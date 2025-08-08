import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import datetime
import uuid
import tempfile
import os

# Paths to TrueType fonts (regular and bold) for PDF
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# -------- Calculation Functions --------

def calculate_es(ATB: float, ATK: float) -> float:
    try:
        return round(100 * (1 - ATB / ATK), 2)
    except ZeroDivisionError:
        return None

def calculate_granulation(init_mass, r02, r04):
    if init_mass <= 0:
        return None, None, None
    p02 = round(100 * r02 / init_mass, 2)
    p04 = round(100 * r04 / init_mass, 2)
    pass04 = round(100 * (init_mass - r04) / init_mass, 2)
    return p02, p04, pass04

def calculate_humidity(wet, dry):
    try:
        return round(100 * (wet - dry) / dry, 2)
    except ZeroDivisionError:
        return None

def calculate_mb_index(volume_ml, conc_g_per_l, sample_g):
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

# -------- Technical Analysis Function --------

def sand_technical_analysis(sample: dict) -> (str, str):
    """
    Analyze sand quality against typical mortar sand norms.
    Returns (verdict, detailed_analysis)
    """
    verdicts = []
    # Typical norms (adapt as needed)
    norms = {
        "Granulation 0–0.2 mm (%)": (0, 20),    # Fine fraction, not too high
        "Granulation 0.2–0.4 mm (%)": (10, 70), # Main fraction
        "Passing <0.4 mm (%)": (60, 100),       # Fines for mortar
        "Humidity (%)": (0, 5),                 # Should be low for storage
        "MB Index (g/100 g)": (0, 1.5),         # Low clay content
        "ES (%)": (75, 100),                    # Cleanliness
    }

    verdict = "PASS"
    for key, (low, high) in norms.items():
        val = sample.get(key)
        if val is None:
            verdicts.append(f"{key}: Not measured.")
            verdict = "FAIL"
            continue
        if val < low:
            verdicts.append(f"{key} below norm ({val} < {low})")
            verdict = "FAIL"
        elif val > high:
            verdicts.append(f"{key} above norm ({val} > {high})")
            verdict = "FAIL"
        else:
            verdicts.append(f"{key}: OK ({val})")

    explanation = (
        "Technical Analysis against mortar sand norms:\n" +
        "\n".join(verdicts) + "\n\n"
    )
    if verdict == "PASS":
        explanation += (
            "The sand sample meets all standard requirements for use in mortar production. "
            "It is suitable for high-quality mortar mixes."
        )
    else:
        explanation += (
            "The sand sample does NOT meet all requirements for standard mortar. "
            "Refer to the above parameter(s) for details."
        )
    return verdict, explanation

# -------- PDF Generation --------

def generate_pdf(sample_data: dict, qr_img: Image.Image) -> bytes:
    """
    Build PDF report embedding QR code and using a Unicode font.
    Now includes a technical analysis section.
    """
    pdf = FPDF()
    pdf.add_page()

    # Register regular and bold fonts
    pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
    pdf.add_font("DejaVu", "B", FONT_PATH_BOLD, uni=True)

    # Header
    pdf.set_font("DejaVu", size=12)
    pdf.cell(0, 10, "Clay & Sand Test Report", ln=1, align='C')
    pdf.ln(2)
    pdf.set_font("DejaVu", size=10)
    pdf.cell(0, 8, f"Sample ID: {sample_data['Sample ID']}", ln=1)
    pdf.cell(0, 8, f"Date: {sample_data['Date']}", ln=1)
    pdf.ln(5)

    # Technical analysis
    verdict, analysis = sand_technical_analysis(sample_data)
    pdf.set_font("DejaVu", size=11, style="B")
    pdf.cell(0, 8, f"Technical Verdict: {verdict}", ln=1)
    pdf.set_font("DejaVu", size=9)
    for line in analysis.splitlines():
        pdf.multi_cell(0, 6, line)
    pdf.ln(4)

    # Embed QR via a temporary PNG file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        qr_img.save(tmp, format="PNG")
        tmp.flush()
        pdf.image(tmp.name, x=160, y=20, w=30)
    finally:
        tmp.close()
        os.unlink(tmp.name)

    pdf.ln(10)
    pdf.set_font("DejaVu", size=11)
    pdf.cell(0, 8, "Test Results:", ln=1)
    pdf.set_font("DejaVu", size=10)
    # Write out each field
    for key, val in sample_data.items():
        if key in ("Sample ID", "Date"):
            continue
        pdf.cell(0, 8, f"{key}: {val}", ln=1)

    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    elif isinstance(pdf_bytes, bytes):
        pass
    else:
        pdf_bytes = bytes(pdf_bytes)
    return pdf_bytes

# -------- Session State --------

if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame()

# -------- App UI --------

st.title("🧪 Clay & Sand Testing Dashboard")

st.markdown("## 📋 Step-by-Step Protocol")
st.markdown("""
1. **Sample Prep**  
   • Weigh ~100 g dry sample and record “Initial Mass.”  
2. **Equivalent Sand (ES)**  
   • Mix sample + kaolin control in water, measure turbidity after 10 min.  
   • ES = 100 × (1 – ATB/ATK).  
3. **Granulation**  
   • Sieve through 0.2 mm & 0.4 mm, weigh retained fractions.  
   • % Retained = (mass_retained / init_mass) × 100.  
4. **Humidity**  
   • Weigh wet, dry at 105 °C to constant mass.  
   • Humidity % = (wet – dry)/dry × 100.  
5. **Methylene Blue**  
   • Add MB solution dropwise until faint ring appears on filter paper.  
   • MB Index = (vol mL × conc g/L ÷ 1000) / sample g × 100.
""")

with st.sidebar:
    st.header("🔧 Inputs")

    sample_id = st.text_input("Sample ID", str(uuid.uuid4())[:8])
    date = st.date_input("Test Date", datetime.date.today())

    st.subheader("ES Test")
    atb = st.number_input("ATB (NTU)", value=50.0, step=0.1)
    atk = st.number_input("ATK (NTU)", value=200.0, step=0.1)

    st.subheader("Granulation")
    init_mass = st.number_input("Initial Mass (g)", value=100.0, step=0.1)
    r02 = st.number_input("Retained 0–0.2 mm (g)", value=30.0, step=0.1)
    r04 = st.number_input("Retained 0.2–0.4 mm (g)", value=50.0, step=0.1)

    st.subheader("Humidity")
    wet = st.number_input("Wet Mass (g)", value=105.0, step=0.1)
    dry = st.number_input("Dry Mass (g)", value=100.0, step=0.1)

    st.subheader("Methylene Blue")
    mb_vol = st.number_input("Volume (mL)", value=10.0, step=0.1)
    mb_conc = st.number_input("Conc. (g/L)", value=1.95, step=0.01)
    mb_g = st.number_input("Sample Mass (g)", value=10.0, step=0.1)

    st.subheader("Observations")
    color = st.text_input("Color")
    odor = st.text_input("Odor")
    notes = st.text_area("Notes")

    run = st.button("🧪 Run & Save")

if run:
    es   = calculate_es(atb, atk)
    p02, p04, p_pass = calculate_granulation(init_mass, r02, r04)
    hum  = calculate_humidity(wet, dry)
    mb_i = calculate_mb_index(mb_vol, mb_conc, mb_g)

    rec = {
        "Sample ID": sample_id,
        "Date": date.isoformat(),
        "ES (%)": es,
        "Granulation 0–0.2 mm (%)": p02,
        "Granulation 0.2–0.4 mm (%)": p04,
        "Passing <0.4 mm (%)": p_pass,
        "Humidity (%)": hum,
        "MB Index (g/100 g)": mb_i,
        "Color": color,
        "Odor": odor,
        "Notes": notes
    }

    qr_str = f"{sample_id}|{date}|ES={es}|MB={mb_i}"
    qr_img = generate_qr(qr_str)

    st.session_state.history = pd.concat(
        [st.session_state.history, pd.DataFrame([rec])],
        ignore_index=True
    )
    st.session_state.last   = rec
    st.session_state.last_qr = qr_img
    st.success(f"Saved Sample {sample_id}")

if not st.session_state.history.empty:
    st.subheader("📒 Batch History")
    st.dataframe(st.session_state.history)

    csv = st.session_state.history.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "batch_history.csv", "text/csv")

    if "last" in st.session_state:
        st.subheader(f"📄 Last Sample: {st.session_state.last['Sample ID']}")
        buf = BytesIO()
        st.session_state.last_qr.save(buf, format="PNG")
        st.image(buf.getvalue(), width=100, caption="QR Code")

        if st.button("📑 Export PDF"):
            pdf_bytes = generate_pdf(st.session_state.last, st.session_state.last_qr)
            if isinstance(pdf_bytes, str):
                pdf_bytes = pdf_bytes.encode("latin-1")
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes,
                file_name=f"Report_{st.session_state.last['Sample ID']}.pdf",
                mime="application/pdf"
            )
