from fpdf import FPDF
from io import BytesIO

def generate_pdf(name, age, sex, complaint, predictions, annotated_img):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    pdf.cell(0, 10, "AffoDent Dental Screening Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Patient Name: {name}", ln=True)
    pdf.cell(0, 8, f"Age: {age}", ln=True)
    pdf.cell(0, 8, f"Sex: {sex}", ln=True)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Chief Complaint / Notes: {complaint}")
    pdf.ln(10)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "AI Detected Findings:", ln=True)
    pdf.set_font("Arial", '', 12)

    if len(predictions) == 0:
        pdf.cell(0, 8, "No significant dental issues detected.", ln=True)
    else:
        for i, (box, score, label, mask) in enumerate(predictions, 1):
            pdf.cell(0, 8, f"Issue {i}: Label {label}, Confidence: {score:.2f}", ln=True)

    pdf.ln(10)
    pdf.cell(0, 10, "Annotated Image:", ln=True)

    # Save annotated image to buffer
    buf = BytesIO()
    annotated_img.save(buf, format='PNG')
    buf.seek(0)
    pdf.image(buf, x=10, w=pdf.w - 20)

    pdf_output = pdf.output(dest='S').encode('latin1')
    return pdf_output
