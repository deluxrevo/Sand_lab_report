# clay_sand_lab_app.py

import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
import datetime
import uuid
from PIL import Image

# ------------ Utility Functions ------------

def calculate_es(ATB: float, ATK: float) -> float:
    """
    Equivalent Sand (ES %) = 100 * (1 - ATB/ATK)
    ATB: turbidity of sample after 10 min (NTU)
    ATK: turbidity of kaolin control (NTU)
    """
    try:
        return round(100 * (1 - ATB / ATK), 2)
    except ZeroDivisionError:
        return None

def calculate_granulation(initial_mass: float,
                          retained_0_2: float,
                          retained_0_4: float):
    """
    Returns three values:
      - % retained on 0–0.2 mm
      - % retained on 0.2–0.4 mm
      - % passing < 0.4 mm
    """
    if initial_mass <= 0:
        return None, None, None
    p02 = round(100 * retained_0_2 / initial_mass, 2)
    p04 = round(100 * retained_0_4 / initial_mass, 2)
    pass04 = round(100 * (initial_mass - retained_0_4) / initial_mass, 2)
    return p02, p04, pass04

def calculate_humidity(wet_mass: float, dry_mass: float) -> float:
    """
    Humidity (%) = 100 * (wet_mass - dry_mass) / dry_mass
    """
    try:
        return round(100 * (wet_mass - dry_mass) / dry_mass, 2)
    except ZeroDivisionError:
        return None

def generate_qr(data_str: str) -> Image.Image:
    """
    Generate a PIL Image QR code for the given string.
    """
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

def generate_pdf(sample_data: dict, qr_img: Image.Image) -> bytes:
    """
    Create a PDF report for one sample, embedding the QR code.
    Returns raw PDF bytes.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Header
    pdf.cell(0, 10, "Clay & Sand Test Report", ln=1, align='C')
    pdf.ln(2)
    pdf.cell(0, 8, f"Sample ID: {sample_data['Sample ID']}", ln=1)
    pdf.cell(0, 8, f"Date: {sample_data['Date']}", ln=1)
    pdf.ln(5)

    # Embed QR code
    buf = BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    pdf.image(buf, x=160, y=20, w=30)
    pdf.ln(15)

    # Results
    for key, val in sample_data.items():
        if key in ("Sample ID", "Date"):
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

    # Unique Sample ID
    sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
    date = st.date_input("Test Date", value=datetime.date.today())

    st.markdown("**ES Test Inputs**")
    atb = st.number_input("Turbidity After 10 min (NTU)", value=50.0, step=0.1, format="%.1f")
    atk = st.number_input("Turbidity Kaolin Control (NTU)", value=200.0, step=0.1, format="%.1f")

    st.markdown("**Granulation Inputs (g)**")
    init_mass = st.number_input("Initial Sample Mass (g)", value=100.0, step=0.1)
    retained_0_2 = st.number_input("Mass Retained on 0–0.2 mm (g)", value=30.0, step=0.1)
    retained_0_4 = st.number_input("Mass Retained on 0.2–0.4 mm (g)", value=50.0, step=0.1)

    st.markdown("**Humidity Inputs (g)**")
    wet_mass = st.number_input("Wet Sample Mass (g)", value=105.0, step=0.1)
    dry_mass = st.number_input("Dry Sample Mass (g)", value=100.0, step=0.1)

    st.markdown("**Observations**")
    color = st.text_input("Color (visual)")
    odor = st.text_input("Odor")
    notes = st.text_area("Additional Notes")

    run = st.button("🧪 Run & Save Test")

if run:
    # Perform calculations
    es = calculate_es(atb, atk)
    g_0_2, g_0_4, passing_0_4 = calculate_granulation(init_mass, retained_0_2, retained_0_4)
    humidity = calculate_humidity(wet_mass, dry_mass)

    # Prepare record
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

    # Generate QR
    qr_data = f"{sample_id}|{date}|ES={es}%"
    qr_img = generate_qr(qr_data)

    # Append to history
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([record])],
        ignore_index=True
    )
    st.session_state.last_sample = record
    st.session_state.last_qr_img = qr_img

    st.success(f"Sample {sample_id} saved to batch history.")

# Display batch history & export options
if not st.session_state.history.empty:
    st.subheader("📒 Batch History")
    st.dataframe(st.session_state.history)

    csv_data = st.session_state.history.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full History (CSV)",
        data=csv_data,
        file_name="batch_history.csv",
        mime="text/csv"
    )

    if "last_sample" in st.session_state:
        st.subheader(f"📄 Last Test: {st.session_state.last_sample['Sample ID']}")

        # Convert PIL QR image to PNG bytes for Streamlit
        buf = BytesIO()
        st.session_state.last_qr_img.save(buf, format="PNG")
        buf.seek(0)
        st.image(buf.getvalue(), width=100, caption="Sample QR Code")

        if st.button("📑 Export Last Report (PDF)"):
            pdf_bytes = generate_pdf(st.session_state.last_sample, st.session_state.last_qr_img)
            st.download_button(
                label="⬇ Download PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{st.session_state.last_sample['Sample ID']}.pdf",
                mime="application/pdf"
            )
