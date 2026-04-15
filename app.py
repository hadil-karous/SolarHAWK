import streamlit as st
from ultralytics import YOLO # pyright: ignore[reportMissingImports]
from PIL import Image
import pandas as pd
import tempfile
from fpdf import FPDF # pyright: ignore[reportMissingModuleSource]
from utils.recommendations import generate_recommendations

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SolarHawk AI",
    page_icon="☀️",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
import base64
from pathlib import Path

# ---------------- IMAGE TO BASE64 UTILITY ----------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ---------------- CUSTOM CSS ----------------
def load_css():
    st.markdown("""
        <style>
        /* Main background */
        .main { background-color: #f8f9fa; }
        
        /* The Green Header Rectangle */
        .header-container {
            background-color: #2e7d32; /* Agricultural Green */
            padding: 20px;
            border-radius: 12px;
            color: white;
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .header-logo {
            flex: 0 0 100px; /* Fixed width for logo */
            margin-right: 20px;
        }

        .header-text h1 {
            color: white !important;
            margin: 0 !important;
            font-size: 2rem !important;
            border-bottom: none !important;
        }

        .header-text p {
            margin: 2px 0 0 0 !important;
            font-size: 1rem;
            opacity: 0.9;
        }
        </style>
    """, unsafe_allow_html=True)

load_css()

# ---------------- BRANDED HEADER EXECUTION ----------------
try:
    # 1. Convert logo to base64 so it renders perfectly in HTML
    logo_path = "assets/logo.png" 
    logo_base64 = get_base64_of_bin_file(logo_path)

    # 2. Render the Green Rectangle with everything inside
    st.markdown(f"""
        <div class="header-container">
            <div class="header-logo">
                <img src="data:image/png;base64,{logo_base64}" width="100" style="border-radius: 8px;">
            </div>
            <div class="header-text">
                <h1>SolarHawk: Thermal Anomaly Analysis</h1>
                <p><b>Developed by Hadil Karous & Mariem Ben Attia</b></p>
                <p>AI for Smart Solar Agriculture 🌱</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.error("⚠️ Logo not found in 'assets/logo.png'. Please check the file path.")

st.write("Upload a thermal drone image to generate a health report.")

# ---------------- CONFIGURATION ----------------
# Make sure "best.pt" is your newly trained YOLO11 model weights!
MODEL_PATH = "best.pt"

# UPDATED FOR NEW DATASET (4 Classes)
CLASS_METRICS = {
    'Diode anomaly':    {'severity': 'High',     'penalty': 30, 'loss_factor': 0.33},
    'Hot Spots':        {'severity': 'Medium',   'penalty': 15, 'loss_factor': 0.10},
    'Reverse polarity': {'severity': 'Critical', 'penalty': 80, 'loss_factor': 1.00},
    'Vegetation':       {'severity': 'Low',      'penalty': 10, 'loss_factor': 0.05}
}

# ---------------- IMAGE ANALYSIS FUNCTION ----------------
def analyze_image(image, model):

    results = model(image)

    report_data = {
        'total_anomalies': 0,
        'health_score': 100,
        'est_power_loss': 0.0,
        'detections': []
    }

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1
            box_area = width * height
            total_area = image.width * image.height
            affected_area_pct = (box_area / total_area) * 100

            metrics = CLASS_METRICS.get(
                cls_name,
                {'severity': 'Unknown', 'penalty': 0, 'loss_factor': 0}
            )

            report_data['total_anomalies'] += 1
            report_data['health_score'] -= metrics['penalty']
            report_data['est_power_loss'] += (metrics['loss_factor'] * 100)

            report_data['detections'].append({
                'Type': cls_name,
                'Confidence': f"{conf:.2f}",
                'Severity': metrics['severity'],
                'Affected Area': f"{affected_area_pct:.1f}%",
                'Est. Loss': f"{metrics['loss_factor']*100:.0f}%"
            })

    report_data['health_score'] = max(0, report_data['health_score'])
    report_data['est_power_loss'] = min(100.0, report_data['est_power_loss'])

    res_plotted = results[0].plot()
    res_image = Image.fromarray(res_plotted[..., ::-1])

    return report_data, res_image

# ---------------- PDF GENERATION ----------------
def generate_pdf(report_data, image_path, recommendations):

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ------------------ GREEN HEADER RECTANGLE ------------------
    pdf.set_fill_color(11, 61, 145)  # Dark Blue (your theme)
    pdf.rect(0, 0, 210, 40, 'F')  # Full width rectangle (A4 width = 210mm)

    # ------------------ LOGO ------------------
    logo_path = "assets/logo.png"   # change name if needed
    try:
        pdf.image(logo_path, x=10, y=8, w=25)
    except:
        pass # Skip logo if missing during PDF generation

    # ------------------ HEADER TEXT ------------------
    pdf.set_text_color(255, 255, 255)  # White text

    pdf.set_xy(40, 10)
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 8, "SolarHawk AI Inspection Report", ln=True)

    pdf.set_x(40)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 6, "Developed by Hadil Karous & Mariem Ben Attia", ln=True)

    pdf.set_x(40)
    pdf.cell(0, 6, "AI for Smart Solar Agriculture", ln=True)

    # Reset text color to black
    pdf.set_text_color(0, 0, 0)

    pdf.ln(35)

    # ------------------ IMAGE SECTION ------------------
    current_y = pdf.get_y()

    image_width = 170
    page_width = pdf.w
    x_position = (page_width - image_width) / 2

    pdf.image(image_path, x=x_position, y=current_y, w=image_width)

    pdf.set_y(current_y + 180)
    pdf.ln(10)

    # ------------------ SUMMARY BOX ------------------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Inspection Summary", ln=True)

    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 8, f"Health Score: {report_data['health_score']}/100", ln=True)
    pdf.cell(0, 8, f"Estimated Power Loss: {report_data['est_power_loss']}%", ln=True)
    pdf.cell(0, 8, f"Detected Anomalies: {report_data['total_anomalies']}", ln=True)

    pdf.ln(8)

    # ------------------ RECOMMENDATIONS ------------------
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 10, "Recommendations", ln=True)

    pdf.set_font("Arial", "", 11)

    if recommendations:
        for rec in recommendations:
            pdf.multi_cell(0, 8, f"- {rec}")
            pdf.ln(2)
    else:
        pdf.cell(0, 8, "No specific recommendations at this time.", ln=True)

    return pdf

# ---------------- STREAMLIT UI ----------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    model = YOLO(MODEL_PATH)
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Panel Health"):

        with st.spinner("Running YOLO11 + Severity Logic..."):

            report, outcome_img = analyze_image(image, model)

            st.image(outcome_img, caption="Detected Anomalies", use_column_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Health Score", f"{report['health_score']}/100")
            col2.metric("Power Loss", f"{report['est_power_loss']}%")
            col3.metric("Anomalies", report['total_anomalies'])

            if report['detections']:
                df = pd.DataFrame(report['detections'])
                st.dataframe(df)
            else:
                st.success("No anomalies detected. Panel is healthy!")

            # -------- RECOMMENDATIONS --------
            # Ensure generate_recommendations in your utils folder is updated to handle the new classes!
            try:
                recommendations = generate_recommendations(report)
            except Exception as e:
                recommendations = ["Could not generate recommendations. Check utils/recommendations.py."]
                st.warning(f"Recommendation engine error: {e}")

            if recommendations:
                st.subheader("🌱 Recommended Actions for Agricultural Use")
                for rec in recommendations:
                    st.write("•", rec)

            # -------- PDF --------
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                outcome_img.save(tmp_file.name)

                pdf = generate_pdf(report, tmp_file.name, recommendations)
                pdf_output = pdf.output(dest="S").encode("latin-1")

                st.download_button(
                    label="📄 Download Official PDF Report",
                    data=pdf_output,
                    file_name="Solar_Inspection_Report.pdf",
                    mime="application/pdf"
                )