import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision
import numpy as np
import os
import requests
import affodent_report  # Make sure this file is in the same folder

st.set_page_config(page_title="AffoDent Dental Screening", layout="centered")

# --- Auto download model weights if missing ---
def download_weights(url, save_path):
    if not os.path.exists(save_path):
        st.info("Downloading model weights...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = st.progress(0)
        downloaded = 0

        with open(save_path, 'wb') as file:
            for data in response.iter_content(block_size):
                downloaded += len(data)
                file.write(data)
                progress_bar.progress(min(downloaded / total_size_in_bytes, 1.0))

        st.success("Model weights downloaded!")

weights_url = "https://github.com/The-ML-Hero/DentalDiagnosisTools/releases/download/v1.0/MASK_RCNN_ROOT_SEGMENTATION.pth"
weights_path = "models/MASK_RCNN_ROOT_SEGMENTATION.pth"

os.makedirs("models", exist_ok=True)
download_weights(weights_url, weights_path)

# --- Load Mask R-CNN model ---
@st.cache_resource(show_spinner=True)
def load_model(weights_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model

# --- Predict detections ---
def predict(image, model, threshold=0.5):
    img_tensor = torchvision.transforms.functional.to_tensor(image)
    outputs = model([img_tensor])[0]
    boxes = outputs['boxes'].detach().numpy()
    scores = outputs['scores'].detach().numpy()
    labels = outputs['labels'].detach().numpy()
    masks = outputs['masks'].detach().numpy()

    filtered = [(b, s, l, m) for b, s, l, m in zip(boxes, scores, labels, masks) if s > threshold]
    return filtered

# --- Draw boxes on image ---
def draw_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = {1: 'red'}

    for box, score, label, mask in predictions:
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=colors[label], width=3)
        draw.text((x1, y1-12), f"Issue {label} {score:.2f}", fill=colors[label], font=font)
    return image

# --- Streamlit UI ---
st.title("AffoDent Dental Screening Tool")
st.write("Upload a dental X-ray or photo to detect dental issues with AI.")

with st.form("patient_form"):
    name = st.text_input("Patient Name", max_chars=50)
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"])
    complaint = st.text_area("Chief Complaint / Notes", max_chars=500)
    uploaded_file = st.file_uploader("Upload Dental Image (PNG, JPG, JPEG)", type=["png","jpg","jpeg"])
    submitted = st.form_submit_button("Analyze")

if submitted:
    if not name or not uploaded_file:
        st.error("Please enter the patient name and upload an image.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        model = load_model(weights_path)
        predictions = predict(image, model)

        annotated_img = draw_boxes(image.copy(), predictions)
        st.image(annotated_img, caption="Annotated Dental Image", use_column_width=True)

        # Generate PDF report bytes
        pdf_bytes = affodent_report.generate_pdf(name, age, sex, complaint, predictions, annotated_img)

        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"AffoDent_Report_{name.replace(' ', '_')}.pdf",
            mime="application/pdf"
        )
