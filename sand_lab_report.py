import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
import datetime
import uuid
from PIL import Image

# ------------ Utility Functions ------------

def calculate_es(ATB, ATK):
    """
    Equivalent Sand (ES) calculation based on turbidity.
    ATB: Turbidity of sample after 10 min (NTU)
    ATK: Turbidity of kaolin control (NTU)
    ES = 100 * (1 - ATB/ATK)
    """
    try:
        return round(100 * (1 - ATB / ATK), 2)
    except ZeroDivisionError:
        return None

def calculate_granulation(initial_mass, retained_0_2, retained_0_4):
    """
    Returns % retained on 0–0.2 mm, 0.2–0.4 mm, and % passing < 0.4 mm.
    """
    if initial_mass <= 0:
        return None, None, None
    p_0_2 = round(100 * retained_0_2 / initial_mass, 2)
    p_0_4 = round(100 * retained_0_4 / initial_mass, 2)
    passing_0_4 = round(100 * (initial_mass - retained_0_4) / initial_mass, 2)
    return p_0_2, p_0_4, passing_0_4

def calculate_humidity(wet_mass, dry_mass):
    """
    Humidity (%) = 100 * (wet − dry) / dry
    """
    try:
        return round(100 * (wet_mass - dry_mass) / dry_mass, 2)
    except ZeroDivisionError:
        return None

def generate_qr(data_str):
    """
    Returns a PIL image of a QR code encoding data_str.
    """
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def generate_pdf(sample_data, qr_img):
    """
    Builds a PDF report for one sample, embedding the QR code.
    Returns: PDF bytes
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title & Meta
    pdf.cell(0, 10, "Clay & Sand Test Report", ln=1, align='C')
    pdf.ln(2)
    pdf.cell(0, 8, f"Sample ID: {sample_data['Sample ID']}", ln=1)
    pdf.cell(0, 8, f"Date: {sample_data['Date']}", ln=1)
    pdf.ln(5)

    # Embed QR
    qr_buffer = BytesIO()
    qr_img.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    pdf.image(qr_buffer, x=160, y=20, w=30)

    pdf.ln(10)
    # Test Results
    for key, val in sample_data.items():
        if key in ["Sample ID", "Date"]:
            continue
        pdf.cell(0, 8, f"{key}: {val}", ln=1)

    return pdf.output(dest="S").encode("latin-1")

# ------------ Session State Setup ------------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ------------ App UI ------------

st.title("🧪 Professional Clay & Sand Testing Dashboard")

with st.sidebar:
    st.header("Raw Inputs & Settings")
    # Auto-generate a unique Sample ID if left blank
    sample_id = st.text_input(
        "Sample ID", 
        value=str(uuid.uuid4())[:8]
    )
    date = st.date_input("Test Date", value=datetime.date.today())

    st.markdown("**ES Test Inputs**")
    atb = st.number_input("Turbidity After 10 min (NTU)", value=50.0, step=0.1, format="%.1f")
    atk = st.number_input("Turbidity Kaolin Control (NTU)", value=200.0, step=0.1, format="%.1f")

    st.markdown("**Granulation Inputs (g)**")
    init_mass = st.number_input("Initial Sample Mass", value=100.0, step=0.1, format="%.1f")
    retained_0_2 = st.number_input("Mass Retained on 0–0.2 mm", value=30.0, step=0.1)
    retained_0_4 = st.number_input("Mass Retained on 0.2–0.4 mm", value=50.0, step=0.1)

    st.markdown("**Humidity Inputs (g)**")
    wet_mass = st.number_input("Wet Sample Mass", value=105.0, step=0.1)
    dry_mass = st.number_input("Dry Sample Mass", value=100.0, step=0.1)

    st.markdown("**Observations**")
    color = st.text_input("Color (visual)")
    odor = st.text_input("Odor")
    notes = st.text_area("Additional Notes")

    if st.button("🧪 Run & Save Test"):
        # Perform calculations
        es = calculate_es(atb, atk)
        g_0_2, g_0_4, passing_0_4 = calculate_granulation(init_mass, retained_0_2, retained_0_4)
        humidity = calculate_humidity(wet_mass, dry_mass)

        # Generate QR
        qr_data = f"{sample_id}|{date}|ES={es}%"
        qr_img = generate_qr(qr_data)

        # Build record
        record = {
            "Sample ID": sample_id,
            "Date": date.isoformat(),
            "ES (%)": es,
            "Granulation 0–0.2 mm (%)": g_0_2,
            "Granulation 0.2–0.4 mm (%)": g_0_4,
            "Passing < 0.4 mm (%)": passing_0_4,
            "Humidity (%)": humidity,
            "Color": color,
            "Odor": odor,
            "Notes": notes
        }

        # Append to history
        st.session_state.history = pd.concat([
            st.session_state.history, 
            pd.DataFrame([record])
        ], ignore_index=True)

        st.success(f"Sample {sample_id} saved to batch history.")

        # Store last sample & QR for PDF export
        st.session_state.last_sample = record
        st.session_state.last_qr_img = qr_img

# ------------ Display Batch History & Export Options ------------

if not st.session_state.history.empty:
    st.subheader("📒 Batch History")
    st.dataframe(st.session_state.history)

    csv_bytes = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full History (CSV)",
        data=csv_bytes,
        file_name="batch_history.csv",
        mime="text/csv"
    )

    if "last_sample" in st.session_state:
        st.subheader(f"📄 Last Test: {st.session_state.last_sample['Sample ID']}")
        st.image(st.session_state.last_qr_img, width=100, caption="Sample QR Code")

        if st.button("📑 Export Last Report (PDF)"):
            pdf_bytes = generate_pdf(st.session_state.last_sample, st.session_state.last_qr_img)
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{st.session_state.last_sample['Sample ID']}.pdf",
                mime="application/pdf"
            )