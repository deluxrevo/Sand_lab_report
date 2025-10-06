import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import datetime
import uuid
import os
from dataclasses import dataclass, asdict

# --- Configuration ---

@dataclass
class AppConfig:
    """Holds application configuration and constants."""
    # Paths to TrueType fonts for PDF generation.
    # Ensure these fonts are available on the system where the app is running.
    FONT_PATH: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # Technical norms for sand quality analysis.
    NORMS = {
        "Granulation 0–0.2 mm (%)": (0, 20),
        "Granulation 0.2–0.4 mm (%)": (10, 70),
        "Passing <0.4 mm (%)": (60, 100),
        "Humidity (%)": (0, 5),
        "Estimated Clay Content (%)": (0, 2.5), # Adjusted norm for clay content
        "ES (%)": (75, 100),
    }
    
    # Cation Exchange Capacity (CEC) for different clay minerals in mg of MB per g of clay.
    # This is the basis for converting MBV to a clay percentage.
    MINERAL_CAPACITY = {
        "Smectite": 100,  # High capacity (e.g., Montmorillonite)
        "Illite": 40,      # Medium capacity
        "Kaolinite": 15,   # Low capacity
    }

    @staticmethod
    def check_font_paths():
        """Validate that font files exist."""
        if not os.path.exists(AppConfig.FONT_PATH) or not os.path.exists(AppConfig.FONT_PATH_BOLD):
            st.error(
                "Font files not found. Please ensure 'DejaVuSans.ttf' and 'DejaVuSans-Bold.ttf' "
                f"are at '{os.path.dirname(AppConfig.FONT_PATH)}'. "
                "On Debian/Ubuntu, you can install them with: `sudo apt-get install fonts-dejavu`"
            )
            st.stop()

@dataclass
class SampleData:
    """Holds all input data from the UI for a single sample."""
    sample_id: str
    date: datetime.date
    atb: float
    atk: float
    init_mass: float
    r02: float
    r04: float
    wet_mass: float
    dry_mass: float
    mb_vol: float
    mb_conc_mg_per_ml: float # Concentration in mg/mL
    mb_sample_mass: float
    mb_mineral_type: str
    blend_a_mass: float
    blend_a_clay_pct: float
    blend_b_mass: float
    blend_b_clay_pct: float
    color: str
    odor: str
    notes: str

# --- Calculation & Analysis Logic ---

class SandAnalysis:
    """Contains all calculation and analysis functions for sand testing."""

    @staticmethod
    def calculate_es(atb: float, atk: float) -> float | None:
        """Calculates Equivalent Sand (ES)."""
        if atk == 0:
            return None
        return round(100 * (1 - atb / atk), 2)

    @staticmethod
    def calculate_granulation(init_mass: float, r02: float, r04: float) -> tuple[float | None, float | None, float | None]:
        """Calculates granulation percentages."""
        if init_mass <= 0:
            return None, None, None
        p02 = round(100 * r02 / init_mass, 2)
        p04 = round(100 * r04 / init_mass, 2)
        pass04 = round(100 * (init_mass - r04) / init_mass, 2)
        return p02, p04, pass04

    @staticmethod
    def calculate_humidity(wet: float, dry: float) -> float | None:
        """Calculates humidity percentage."""
        if dry == 0:
            return None
        return round(100 * (wet - dry) / dry, 2)

    @staticmethod
    def calculate_mbv(volume_ml: float, conc_mg_per_ml: float, sample_g: float) -> float | None:
        """Calculates Methylene Blue Value (MBV) in mg/g."""
        if sample_g == 0:
            return None
        # MBV (mg/g) = (Volume of dye in mL * Concentration in mg/mL) / Mass of sample in g
        mbv = (volume_ml * conc_mg_per_ml) / sample_g
        return round(mbv, 2)

    @staticmethod
    def estimate_clay_content(mbv: float, mineral_type: str) -> float | None:
        """Estimates clay content (%) based on MBV and mineral type."""
        if mbv is None:
            return None
        
        mineral_capacity = AppConfig.MINERAL_CAPACITY.get(mineral_type)
        if not mineral_capacity or mineral_capacity == 0:
            return None

        # Clay % = (MBV of sample [mg/g] / Capacity of pure clay [mg/g]) * 100
        clay_percentage = (mbv / mineral_capacity) * 100
        return round(clay_percentage, 2)
    
    @staticmethod
    def calculate_final_clay_percent(a_mass: float, ca_percent: float, b_mass: float, cb_percent: float) -> float | None:
        """Calculates weighted average to estimate final clay percentage."""
        total_mass = a_mass + b_mass
        if total_mass <= 0:
            return None
        final_percent = ((a_mass * ca_percent) + (b_mass * cb_percent)) / total_mass
        return round(final_percent, 2)

    @staticmethod
    def perform_technical_analysis(sample_results: dict) -> tuple[str, str]:
        """Analyzes sand quality against typical mortar sand norms."""
        verdicts = []
        overall_verdict = "PASS"

        for key, (low, high) in AppConfig.NORMS.items():
            val = sample_results.get(key)
            if val is None:
                verdicts.append(f"• {key}: Not measured.")
                overall_verdict = "FAIL"
            elif not (low <= val <= high):
                verdicts.append(f"• {key}: OUT OF NORM (Value: {val}, Expected: {low}-{high})")
                overall_verdict = "FAIL"
            else:
                verdicts.append(f"• {key}: OK (Value: {val})")

        analysis_details = "\n".join(verdicts)
        summary = (
            "The sand sample meets all standard requirements for use in mortar production."
            if overall_verdict == "PASS"
            else "The sand sample does NOT meet all requirements for standard mortar. See details above."
        )
        
        explanation = f"Technical Analysis against mortar sand norms:\n{analysis_details}\n\nSummary: {summary}"
        return overall_verdict, explanation

# --- Utility Functions ---

def generate_qr_code(data_str: str) -> Image.Image:
    """Generates a QR code image from a string."""
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

# --- PDF Generation ---

class PDFReport:
    """Handles the creation of the PDF report."""

    def __init__(self, font_path: str, font_path_bold: str):
        self.pdf = FPDF()
        self.pdf.add_font("DejaVu", "", font_path, uni=True)
        self.pdf.add_font("DejaVu", "B", font_path_bold, uni=True)

    def generate(self, sample_data: dict, qr_img: Image.Image, analysis_verdict: str, analysis_details: str) -> bytes:
        """Builds the complete PDF report."""
        self.pdf.add_page()
        self._add_header(sample_data, qr_img)
        self._add_technical_analysis(analysis_verdict, analysis_details)
        self._add_results_table(sample_data)
        
        # Return PDF as bytes
        return self.pdf.output(dest="S").encode("latin-1")

    def _add_header(self, sample_data: dict, qr_img: Image.Image):
        """Adds the report header, title, and QR code."""
        self.pdf.set_font("DejaVu", "B", size=16)
        self.pdf.cell(0, 10, "Clay & Sand Test Report", ln=True, align='C')
        self.pdf.ln(5)

        # CORRECTED SECTION: Embed QR code from memory
        with BytesIO() as qr_buffer:
            qr_img.save(qr_buffer, format="PNG")
            qr_buffer.seek(0)  # Rewind the buffer to the beginning
            self.pdf.image(qr_buffer, x=160, y=15, w=30, type="PNG") # Explicitly set type

        self.pdf.set_font("DejaVu", size=10)
        self.pdf.cell(0, 8, f"Sample ID: {sample_data['Sample ID']}", ln=True)
        self.pdf.cell(0, 8, f"Date: {sample_data['Date']}", ln=True)
        self.pdf.ln(5)

    def _add_technical_analysis(self, verdict: str, analysis: str):
        """Adds the technical analysis section."""
        self.pdf.set_font("DejaVu", "B", size=12)
        self.pdf.cell(0, 8, f"Technical Verdict: {verdict}", ln=True)
        self.pdf.set_font("DejaVu", size=9)
        self.pdf.multi_cell(0, 5, analysis)
        self.pdf.ln(5)

    def _add_results_table(self, sample_data: dict):
        """Adds the detailed test results."""
        self.pdf.set_font("DejaVu", "B", size=12)
        self.pdf.cell(0, 10, "Test Results", ln=True)
        self.pdf.set_font("DejaVu", "", size=10)
        
        # Set column widths
        self.pdf.set_fill_color(240, 240, 240)
        col_width_key = self.pdf.w * 0.5
        col_width_val = self.pdf.w * 0.3
        
        for key, val in sample_data.items():
            if key in ("Sample ID", "Date"):
                continue
            self.pdf.cell(col_width_key, 8, f"{key}:", border=1)
            self.pdf.cell(col_width_val, 8, str(val if val is not None else "N/A"), border=1, ln=True)

# --- Streamlit UI ---

def initialize_session_state():
    """Initializes session state variables."""
    if "history" not in st.session_state:
        st.session_state.history = pd.DataFrame()
    if "last_run" not in st.session_state:
        st.session_state.last_run = {}

def build_sidebar() -> tuple[SampleData | None, bool]:
    """Creates the sidebar UI for inputs and returns data and run status."""
    with st.sidebar:
        st.header("🔧 Inputs")

        sample_id = st.text_input("Sample ID", f"sample-{uuid.uuid4().hex[:6]}")
        date = st.date_input("Test Date", datetime.date.today())

        with st.expander("ES Test", expanded=True):
            atb = st.number_input("ATB (NTU)", value=50.0, step=0.1, min_value=0.0)
            atk = st.number_input("ATK (NTU)", value=200.0, step=0.1, min_value=0.0)

        with st.expander("Granulation", expanded=True):
            init_mass = st.number_input("Initial Mass (g)", value=100.0, step=0.1, min_value=0.0)
            r02 = st.number_input("Retained 0–0.2 mm (g)", value=30.0, step=0.1, min_value=0.0)
            r04 = st.number_input("Retained 0.2–0.4 mm (g)", value=50.0, step=0.1, min_value=0.0)

        with st.expander("Humidity", expanded=True):
            wet_mass = st.number_input("Wet Mass (g)", value=105.0, step=0.1)
            dry_mass = st.number_input("Dry Mass (g)", value=100.0, step=0.1, min_value=0.0)

        with st.expander("Methylene Blue", expanded=True):
            mb_vol = st.number_input("Volume (mL)", value=1.0, step=0.1, min_value=0.0, help="Volume of MB solution added.")
            # A 1.95% solution is 1.95g/100mL, which is 19.5g/L or 19.5mg/mL.
            mb_conc = st.number_input("Conc. (mg/mL)", value=19.5, step=0.01, min_value=0.0, help="Concentration of the MB solution in milligrams per milliliter.")
            mb_sample_mass = st.number_input("Sample Mass (g)", value=5.0, step=0.1, min_value=0.0)
            mb_mineral = st.selectbox("Assumed Clay Mineral", options=list(AppConfig.MINERAL_CAPACITY.keys()), index=2, help="Select the dominant clay mineral to estimate clay %.")

        with st.expander("Blend Ratio (Final Clay %)", expanded=True):
            blend_a_mass = st.number_input("A: Mass of current sand (g)", value=100.0, step=0.1, min_value=0.0)
            blend_a_clay_pct = st.number_input("C_A: Clay content of current sand (%)", value=2.0, step=0.01, min_value=0.0, max_value=100.0)
            blend_b_mass = st.number_input("B: Mass of clean sand to add (g)", value=0.0, step=0.1, min_value=0.0)
            blend_b_clay_pct = st.number_input("C_B: Clay content of clean sand (%)", value=0.5, step=0.01, min_value=0.0, max_value=100.0)

        with st.expander("Observations"):
            color = st.text_input("Color", "Light Brown")
            odor = st.text_input("Odor", "None")
            notes = st.text_area("Notes", "Standard sample.")

        run_button_pressed = st.button("🧪 Run & Save Analysis", use_container_width=True)
        
        inputs = SampleData(
            sample_id, date, atb, atk, init_mass, r02, r04, wet_mass, dry_mass,
            mb_vol, mb_conc, mb_sample_mass, mb_mineral, blend_a_mass, blend_a_clay_pct,
            blend_b_mass, blend_b_clay_pct, color, odor, notes
        )
        return inputs, run_button_pressed

def build_main_view():
    """Builds the main panel UI for instructions, calculators, and results."""
    st.title("🧪 Clay & Sand Testing Dashboard")

    with st.expander("📋 Step-by-Step Protocol", expanded=False):
        st.markdown("""
        1. **Sample Prep**: Weigh a dry sample and record “Initial Mass.”
        2. **Equivalent Sand (ES)**: Mix sample with kaolin control in water, measure turbidity. `ES = 100 * (1 – ATB/ATK)`.
        3. **Granulation**: Sieve through 0.2 mm & 0.4 mm sieves, then weigh retained fractions.
        4. **Humidity**: Weigh a wet sample, dry at 105°C to constant mass, then weigh again. `Humidity % = (wet – dry)/dry * 100`.
        5. **Methylene Blue**: Titrate with MB solution until a faint ring appears on filter paper. The MBV is calculated and used to estimate clay content based on the assumed mineralogy.
        6. **Blend Ratio**: Estimate final clay % after blending.
        """)
    
    # --- Display History and Last Run ---
    if not st.session_state.history.empty:
        st.subheader("📒 Batch History")
        st.dataframe(st.session_state.history)
        
        csv_data = st.session_state.history.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download History (CSV)", csv_data, "sand_test_history.csv", "text/csv")

    if st.session_state.last_run:
        last_run = st.session_state.last_run
        st.subheader(f"📄 Last Sample: {last_run['results']['Sample ID']}")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(last_run['qr_image'], width=150, caption="Sample QR Code")
            
            pdf_bytes = last_run['pdf_bytes']
            st.download_button(
                "⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"Report_{last_run['results']['Sample ID']}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
        with col2:
            st.metric("Technical Verdict", last_run['verdict'])
            st.json(last_run['results'])


def process_new_sample(inputs: SampleData):
    """Performs all calculations for a new sample and updates session state."""
    es = SandAnalysis.calculate_es(inputs.atb, inputs.atk)
    p02, p04, p_pass = SandAnalysis.calculate_granulation(inputs.init_mass, inputs.r02, inputs.r04)
    hum = SandAnalysis.calculate_humidity(inputs.wet_mass, inputs.dry_mass)
    mbv = SandAnalysis.calculate_mbv(inputs.mb_vol, inputs.mb_conc_mg_per_ml, inputs.mb_sample_mass)
    estimated_clay = SandAnalysis.estimate_clay_content(mbv, inputs.mb_mineral_type)
    final_clay = SandAnalysis.calculate_final_clay_percent(inputs.blend_a_mass, inputs.blend_a_clay_pct, inputs.blend_b_mass, inputs.blend_b_clay_pct)

    results = {
        "Sample ID": inputs.sample_id,
        "Date": inputs.date.isoformat(),
        "ES (%)": es,
        "Granulation 0–0.2 mm (%)": p02,
        "Granulation 0.2–0.4 mm (%)": p04,
        "Passing <0.4 mm (%)": p_pass,
        "Humidity (%)": hum,
        "MBV (mg/g)": mbv,
        "Assumed Mineral": inputs.mb_mineral_type,
        "Estimated Clay Content (%)": estimated_clay,
        "Final Blended Clay (%)": final_clay,
        "Color": inputs.color,
        "Odor": inputs.odor,
        "Notes": inputs.notes,
    }
    
    # Perform technical analysis
    verdict, analysis_details = SandAnalysis.perform_technical_analysis(results)

    # Generate QR code and PDF
    qr_str = f"ID:{inputs.sample_id}|Date:{results['Date']}|ES:{es}|MBV:{mbv}|ClayEst:{estimated_clay}"
    qr_img = generate_qr_code(qr_str)
    
    pdf_generator = PDFReport(AppConfig.FONT_PATH, AppConfig.FONT_PATH_BOLD)
    pdf_bytes = pdf_generator.generate(results, qr_img, verdict, analysis_details)

    # Update history
    new_record_df = pd.DataFrame([results])
    st.session_state.history = pd.concat([st.session_state.history, new_record_df], ignore_index=True)
    
    # Store all artifacts from the last run
    st.session_state.last_run = {
        "results": results,
        "qr_image": qr_img,
        "pdf_bytes": pdf_bytes,
        "verdict": verdict,
        "analysis_details": analysis_details
    }
    st.success(f"✅ Analysis for sample '{inputs.sample_id}' completed and saved.")
    st.rerun()


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    AppConfig.check_font_paths()
    initialize_session_state()
    
    inputs, run_pressed = build_sidebar()
    
    if run_pressed and inputs:
        process_new_sample(inputs)
    
    build_main_view()


if __name__ == "__main__":
    main()
