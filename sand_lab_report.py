# clay_sand_lab_app.py

import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
import datetime
import uuid
from PIL import Image

# ------------ Calculation Functions ------------

def calculate_es(atb, atk):
    """
    Equivalent Sand (ES %):
    ES = 100 * (1 - ATB / ATK)
    where ATB = turbidity after 10 min (sample),
          ATK = turbidity after 10 min (kaolin control).
    """
    try:
        return round(100 * (1 - atb / atk), 2)
    except ZeroDivisionError:
        return None

def calculate_granulation(init_mass, r02, r04):
    """
    Granulation distribution:
      % Retained on 0–0.2 mm = 100 * r02 / init_mass
      % Retained on 0.2–0.4 mm = 100 * r04 / init_mass
      % Passing < 0.4 mm = 100 * (init_mass - r04) / init_mass
    """
    if init_mass <= 0:
        return None, None, None
    p02 = round(100 * r02 / init_mass, 2)
    p04 = round(100 * r04 / init_mass, 2)
    pass04 = round(100 * (init_mass - r04) / init_mass, 2)
    return p02, p04, pass04

def calculate_humidity(wet, dry):
    """
    Humidity (%):
    = 100 * (wet_mass - dry_mass) / dry_mass
    """
    try:
        return round(100 * (wet - dry) / dry, 2)
    except ZeroDivisionError:
        return None

def calculate_mb_index(volume_ml, conc_g_per_l, sample_g):
    """
    Methylene Blue Index (g MB / 100 g sample):
    MB_index = (volume_ml * conc_g_per_l / 1000) / sample_g * 100
    """
    try:
        mb_grams = volume_ml * conc_g_per_l / 1000
        return round(mb_grams / sample_g * 100, 2)
    except ZeroDivisionError:
        return None

def generate_qr(data_str):
    """Return a PIL Image QR code for the given data string."""
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return img

def generate_pdf(sample_data, qr_img):
    """Generate a PDF report embedding the QR code."""
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

    # Results table
    for k, v in sample_data.items():
        if k in ("Sample ID", "Date"):
            continue
        pdf.cell(0, 8, f"{k}: {v}", ln=1)

    return pdf.output(dest="S").encode("latin-1")


# ------------ Session State Initialization ------------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# ------------ Main App UI ------------

st.title("🧪 Clay & Sand Testing Dashboard")

# Step‐by‐step protocol section
st.markdown("## 📋 Step-by-Step Lab Protocol")
st.markdown("""
1. **Sample Prep**  
   • Weigh out ~100 g dry material.  
   • Record as “Initial Sample Mass.”
2. **Equivalent Sand Test (ES)**  
   • Prepare kaolin control (ATK) and sample (ATB).  
   • Measure turbidity after 10 min (NTU) with turbidity meter.  
   • ES calculation: \\(100 \\times (1 - ATB/ATK)\\).
3. **Granulation (Sieving)**  
   • Sieve sample through 0.2 mm and 0.4 mm.  
   • Weigh retained fractions.  
   • Calculate % retained/passing with formulas in code.
4. **Humidity**  
   • Weigh wet sample (after mixing with water).  
   • Dry at 105 °C until constant mass.  
   • Humidity (%) = \\(100 \\times (wet - dry)/dry\\).
5. **Methylene Blue Test**  
   • Add known-concentration MB solution dropwise until a faint blue ring persists.  
   • Record volume added (mL).  
   • MB Index (g/100 g) = \\((vol_{mL} \\times conc_{g/L}/1000)/mass_{g} \\times 100\\).
""")

# Sidebar inputs grouped in expanders
with st.sidebar:
    st.header("🔧 Inputs & Real-Time Metrics")

    with st.expander("ES Test"):
        atb = st.number_input("ATB (Sample Turbidity NTU)", value=50.0, step=0.1)
        atk = st.number_input("ATK (Kaolin Control NTU)", value=200.0, step=0.1)
        es = calculate_es(atb, atk)
        st.metric("Equivalent Sand (ES %)", es or "–")

    with st.expander("Granulation"):
        init_mass = st.number_input("Initial Mass (g)", value=100.0, step=0.1)
        r02 = st.number_input("Retained 0–0.2 mm (g)", value=30.0, step=0.1)
        r04 = st.number_input("Retained 0.2–0.4 mm (g)", value=50.0, step=0.1)
        p02, p04, pass04 = calculate_granulation(init_mass, r02, r04)
        st.metric("0–0.2 mm Retained (%)", p02 or "–")
        st.metric("0.2–0.4 mm Retained (%)", p04 or "–")
        st.metric("<0.4 mm Passing (%)", pass04 or "–")

    with st.expander("Humidity"):
        wet = st.number_input("Wet Mass (g)", value=105.0, step=0.1)
        dry = st.number_input("Dry Mass (g)", value=100.0, step=0.1)
        hum = calculate_humidity(wet, dry)
        st.metric("Humidity (%)", hum or "–")

    with st.expander("Methylene Blue Test"):
        mb_vol = st.number_input("MB Volume Added (mL)", value=10.0, step=0.1)
        mb_conc = st.number_input("MB Solution Conc. (g/L)", value=0.1, step=0.01)
        mb_sample = st.number_input("Sample Mass for MB (g)", value=1.0, step=0.1)
        mb_idx = calculate_mb_index(mb_vol, mb_conc, mb_sample)
        st.metric("MB Index (g/100 g)", mb_idx or "–")

    with st.expander("Observations"):
        color = st.text_input("Color (visual)")
        odor = st.text_input("Odor")
        notes = st.text_area("Additional Notes")

    sample_id = st.text_input("Sample ID", value=str(uuid.uuid4())[:8])
    date = st.date_input("Test Date", value=datetime.date.today())

    run = st.button("🧪 Run & Save Test")


# Process inputs on button click
if run:
    # Build record dict
    record = {
        "Sample ID": sample_id,
        "Date": date.isoformat(),
        "ES (%)": es,
        "Retained 0–0.2 mm (%)": p02,
        "Retained 0.2–0.4 mm (%)": p04,
        "Passing <0.4 mm (%)": pass04,
        "Humidity (%)": hum,
        "MB Index (g/100 g)": mb_idx,
        "Color": color,
        "Odor": odor,
        "Notes": notes
    }

    # Generate QR & append to history
    qr_data = f"{sample_id}|{date}|ES={es}%|MB={mb_idx}"
    qr_img = generate_qr(qr_data)
    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([record])],
        ignore_index=True
    )
    st.session_state.last_sample = record
    st.session_state.last_qr_img = qr_img
    st.success(f"Sample {sample_id} saved to history.")

# Display Batch History & Export
if not st.session_state.history.empty:
    st.subheader("📒 Batch History")
    st.dataframe(st.session_state.history)

    csv_bytes = st.session_state.history.to_csv(index=False).encode()
    st.download_button(
        "⬇ Download Full History (CSV)",
        data=csv_bytes,
        file_name="batch_history.csv",
        mime="text/csv"
    )

    if "last_sample" in st.session_state:
        st.subheader(f"📄 Last Report: {st.session_state.last_sample['Sample ID']}")
        st.image(st.session_state.last_qr_img, width=100, caption="Sample QR Code")

        if st.button("📑 Export PDF Report"):
            pdf_bytes = generate_pdf(st.session_state.last_sample, st.session_state.last_qr_img)
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes,
                file_name=f"Report_{st.session_state.last_sample['Sample ID']}.pdf",
                mime="application/pdf"
            )
