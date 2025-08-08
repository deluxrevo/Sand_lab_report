# clay_sand_lab_app.py

import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
import datetime
import uuid
from PIL import Image

# -------- Calculation Functions --------

def calculate_es(ATB: float, ATK: float) -> float:
    """
    Equivalent Sand (ES %):
    ES = 100 * (1 - ATB/ATK).
    """
    try:
        return round(100 * (1 - ATB / ATK), 2)
    except ZeroDivisionError:
        return None

def calculate_granulation(init_mass, r02, r04):
    """
    Granulation distribution:
      % retained 0–0.2 mm,
      % retained 0.2–0.4 mm,
      % passing <0.4 mm.
    """
    if init_mass <= 0:
        return None, None, None
    p02 = round(100 * r02 / init_mass, 2)
    p04 = round(100 * r04 / init_mass, 2)
    pass04 = round(100 * (init_mass - r04) / init_mass, 2)
    return p02, p04, pass04

def calculate_humidity(wet, dry):
    """
    Humidity (%) = 100 * (wet - dry) / dry.
    """
    try:
        return round(100 * (wet - dry) / dry, 2)
    except ZeroDivisionError:
        return None

def calculate_mb_index(volume_ml, conc_g_per_l, sample_g):
    """
    Methylene Blue Index (g MB / 100 g sample):
    = (volume_ml * conc_g_per_l/1000) / sample_g * 100
    """
    try:
        mb_g = volume_ml * (conc_g_per_l / 1000)
        return round(mb_g / sample_g * 100, 2)
    except ZeroDivisionError:
        return None

def generate_qr(data_str: str) -> Image.Image:
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

def generate_pdf(sample_data: dict, qr_img: Image.Image) -> bytes:
    """
    Build PDF report embedding QR image.
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

    # Embed QR
    buf = BytesIO()
    qr_img.save(buf, format="PNG")
    buf.seek(0)
    pdf.image(buf, x=160, y=20, w=30)
    pdf.ln(15)

    # Results
    for k, v in sample_data.items():
        if k in ("Sample ID", "Date"):
            continue
        pdf.cell(0, 8, f"{k}: {v}", ln=1)

    return pdf.output(dest="S").encode("latin-1")


# -------- Session State --------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# -------- Main App --------

st.title("🧪 Clay & Sand Testing Dashboard")

# Step-by-step protocol
st.markdown("## 📋 Step-by-Step Protocol")
st.markdown("""
1. **Sample Prep**  
   • Weigh ~100 g dry sample.  
   • Record “Initial Sample Mass.”

2. **Equivalent Sand (ES) Test**  
   • Prepare sample and kaolin control in water.  
   • Measure turbidity after 10 min: ATB (sample), ATK (control).  
   • ES = 100 × (1 – ATB/ATK).

3. **Granulation (Sieving)**  
   • Sieve through 0.2 mm & 0.4 mm.  
   • Weigh retained fractions.  
   • % Retained = (mass_retained / initial_mass) × 100.

4. **Humidity**  
   • Weigh wet sample.  
   • Dry at 105 °C to constant mass.  
   • Humidity (%) = (wet – dry)/dry × 100.

5. **Methylene Blue Test**  
   • Pipette MB solution (known concentration) dropwise into sand-water mix on filter paper until faint blue ring remains.  
   • Record volume added (mL).  
   • MB Index (g/100 g) = (vol_mL × conc_g/L ÷ 1000) / sample_g × 100.
""")

with st.sidebar:
    st.header("🔧 Inputs")

    # Sample meta
    sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
    date = st.date_input("Test Date", value=datetime.date.today())

    # ES inputs
    st.subheader("ES Test")
    atb = st.number_input("Sample Turbidity ATB (NTU)", value=50.0, step=0.1)
    atk = st.number_input("Control Turbidity ATK (NTU)", value=200.0, step=0.1)

    # Granulation
    st.subheader("Granulation")
    init_mass = st.number_input("Initial Mass (g)", value=100.0, step=0.1)
    r02 = st.number_input("Retained 0–0.2 mm (g)", value=30.0, step=0.1)
    r04 = st.number_input("Retained 0.2–0.4 mm (g)", value=50.0, step=0.1)

    # Humidity
    st.subheader("Humidity")
    wet = st.number_input("Wet Mass (g)", value=105.0, step=0.1)
    dry = st.number_input("Dry Mass (g)", value=100.0, step=0.1)

    # Methylene Blue
    st.subheader("Methylene Blue")
    mb_vol = st.number_input("MB Volume (mL)", value=10.0, step=0.1)
    mb_conc = st.number_input("MB Conc. (g/L)", value=1.95, step=0.01)
    mb_g_sample = st.number_input("MB Sample Mass (g)", value=10.0, step=0.1)

    # Observations
    st.subheader("Observations")
    color = st.text_input("Color")
    odor = st.text_input("Odor")
    notes = st.text_area("Notes")

    run = st.button("🧪 Run & Save")

if run:
    # Calculate metrics
    es = calculate_es(atb, atk)
    p02, p04, pass04 = calculate_granulation(init_mass, r02, r04)
    hum = calculate_humidity(wet, dry)
    mb_idx = calculate_mb_index(mb_vol, mb_conc, mb_g_sample)

    # Record
    rec = {
        "Sample ID": sample_id,
        "Date": date.isoformat(),
        "ES (%)": es,
        "Granulation 0–0.2 mm (%)": p02,
        "Granulation 0.2–0.4 mm (%)": p04,
        "Passing <0.4 mm (%)": pass04,
        "Humidity (%)": hum,
        "MB Index (g/100 g)": mb_idx,
        "Color": color,
        "Odor": odor,
        "Notes": notes
    }

    qr_str = f"{sample_id}|{date}|ES={es}|MB={mb_idx}"
    qr_img = generate_qr(qr_str)

    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([rec])],
        ignore_index=True
    )
    st.session_state.last = rec
    st.session_state.last_qr = qr_img
    st.success(f"Saved sample {sample_id}")

# Display history & exports
if not st.session_state.history.empty:
    st.subheader("📒 Batch History")
    st.dataframe(st.session_state.history)

    csv = st.session_state.history.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "batch_history.csv", "text/csv")

    if "last" in st.session_state:
        st.subheader(f"📄 Last Sample: {st.session_state.last['Sample ID']}")

        # Render QR from PIL to bytes
        buf_qr = BytesIO()
        st.session_state.last_qr.save(buf_qr, format="PNG")
        st.image(buf_qr.getvalue(), width=100, caption="QR Code")

        if st.button("📑 Export PDF"):
            pdf_bytes = generate_pdf(st.session_state.last, st.session_state.last_qr)
            st.download_button(
                "⬇ Download PDF", pdf_bytes,
                f"Report_{st.session_state.last['Sample ID']}.pdf",
                "application/pdf"
            )

