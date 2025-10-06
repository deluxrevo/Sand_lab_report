import streamlit as st
import pandas as pd
import qrcode
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import datetime
import uuid
import os
from dataclasses import dataclass, field
import tempfile
from typing import List, Dict, Any, Tuple

# --- Configuration ---

@dataclass
class AppConfig:
    """Holds application configuration and constants."""
    FONT_PATH: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    FONT_PATH_BOLD: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    MINERAL_CAPACITY: Dict[str, int] = field(default_factory=lambda: {
        "Smectite": 250, "Illite": 80, "Kaolinite": 25,
    })

    PRODUCT_PROFILES: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "Self-leveling": {
            "Passing <0.4 mm (%)": (70, 100),
            "Fines (<0.063 mm) (%)": (8, 15),
            "Whole-Sand Clay Content (%)": (0, 2.0),
            "ES (%)": (75, 100),
        },
        "Tile adhesive": {
            "Passing <0.4 mm (%)": (60, 95),
            "Fines (<0.063 mm) (%)": (10, 18),
            "Whole-Sand Clay Content (%)": (0, 2.5),
            "ES (%)": (75, 100),
        },
        "Projected/render mortar": {
            "Passing <0.4 mm (%)": (40, 80),
            "Fines (<0.063 mm) (%)": (6, 14),
            "Whole-Sand Clay Content (%)": (0, 3.0),
            "ES (%)": (75, 100),
        },
    })

    @staticmethod
    def check_font_paths():
        if not os.path.exists(AppConfig.FONT_PATH) or not os.path.exists(AppConfig.FONT_PATH_BOLD):
            st.error(f"Font files not found. On Debian/Ubuntu: `sudo apt-get install fonts-dejavu`")
            st.stop()

@dataclass
class SampleData:
    """Holds all input data from the UI for a single sample."""
    product_profile: str; sample_id: str; date: datetime.date
    init_mass: float; r02_mass: float; r04_mass: float; fines_pct: float
    atb: float; atk: float;
    mb_vol: float; mb_conc_mg_per_ml: float; mb_sample_mass: float
    notes: str

@dataclass
class AnalysisResult:
    """Holds all calculated results and verdict information."""
    results_dict: Dict[str, Any]
    verdict: str # RED, AMBER, GREEN
    reason_codes: List[str]

# --- Calculation & Analysis Logic ---

class SandAnalysis:
    """Contains all calculation and analysis functions for sand testing."""

    @staticmethod
    def perform_full_analysis(inputs: SampleData, config: AppConfig) -> AnalysisResult:
        results = {}
        reason_codes = []

        if inputs.init_mass <= 0:
            reason_codes.append("INTEGRITY_FAIL: Initial Mass must be > 0.")
        if inputs.r02_mass + inputs.r04_mass > inputs.init_mass:
            reason_codes.append("INTEGRITY_FAIL: Retained mass cannot exceed Initial Mass.")
        if not (0 <= inputs.fines_pct <= 100):
            reason_codes.append("INTEGRITY_FAIL: Fines must be between 0 and 100%.")
        
        if reason_codes:
            return AnalysisResult(results, "RED", reason_codes)

        p04_mass_retained_as_pct = round(100 * inputs.r04_mass / inputs.init_mass, 2)
        passing_04_pct = round(100 - p04_mass_retained_as_pct, 2)

        results["Passing <0.4 mm (%)"] = passing_04_pct
        results["Fines (<0.063 mm) (%)"] = inputs.fines_pct
        
        es_pct = SandAnalysis._calculate_es(inputs.atb, inputs.atk)
        results["ES (%)"] = es_pct

        mbv = SandAnalysis._calculate_mbv(inputs.mb_vol, inputs.mb_conc_mg_per_ml, inputs.mb_sample_mass)
        results["MBV (mg/g)"] = mbv

        clay_range = {}
        for mineral, capacity in config.MINERAL_CAPACITY.items():
            clay_range[mineral] = SandAnalysis._estimate_whole_sand_clay(mbv, capacity, inputs.fines_pct)
        
        results["Clay Content Range (%)"] = {k: v for k, v in clay_range.items()}
        worst_case_clay = clay_range.get("Kaolinite")
        results["Whole-Sand Clay Content (%)"] = worst_case_clay

        profile_norms = config.PRODUCT_PROFILES.get(inputs.product_profile, {})
        for key, (low, high) in profile_norms.items():
            val = results.get(key)
            if val is not None and not (low <= val <= high):
                reason_codes.append(f"SPEC_FAIL: '{key}' ({val}) is outside the profile norm ({low}-{high}).")

        verdict = "AMBER" if reason_codes else "GREEN"
        return AnalysisResult(results, verdict, reason_codes)

    @staticmethod
    def _calculate_es(atb: float, atk: float) -> float | None:
        if atk <= 0: return None
        return round(100 * (1 - atb / atk), 2)
    
    @staticmethod
    def _calculate_mbv(volume_ml: float, conc_mg_per_ml: float, sample_g: float) -> float | None:
        if sample_g <= 0: return None
        return round((volume_ml * conc_mg_per_ml) / sample_g, 2)

    @staticmethod
    def _estimate_whole_sand_clay(mbv: float, mineral_capacity: int, fines_percent: float) -> float | None:
        if mbv is None or fines_percent is None or mineral_capacity <= 0: return None
        clay_in_fines = (mbv / mineral_capacity) * 100
        whole_sand_clay = clay_in_fines * (fines_percent / 100.0)
        return round(whole_sand_clay, 2)

# --- Utility & PDF ---

def generate_qr_code(data_str: str) -> Image.Image:
    qr = qrcode.QRCode(box_size=4, border=1)
    qr.add_data(data_str)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white")

class PDFReport:
    """Handles the creation of the PDF report."""
    def __init__(self, font_path: str, font_path_bold: str):
        self.pdf = FPDF()
        self.pdf.add_font("DejaVu", "", font_path, uni=True)
        self.pdf.add_font("DejaVu", "B", font_path_bold, uni=True)

    def generate(self, sample_info: dict, analysis_result: AnalysisResult, qr_image_path: str) -> BytesIO:
        self.pdf.add_page()
        self._add_header(sample_info, qr_image_path)
        self._add_verdict_section(analysis_result)
        self._add_results_table(analysis_result.results_dict)
        return BytesIO(self.pdf.output())

    def _add_header(self, sample_info: dict, qr_image_path: str):
        self.pdf.set_font("DejaVu", "B", 16)
        self.pdf.cell(0, 10, "Sand Quality Control Report", ln=True, align='C')
        self.pdf.image(qr_image_path, x=self.pdf.w - 40, y=10, w=30)
        self.pdf.set_font("DejaVu", "", 10)
        self.pdf.cell(0, 8, f"Sample ID: {sample_info.get('sample_id')}", ln=True)
        self.pdf.cell(0, 8, f"Date: {sample_info.get('date').isoformat()}", ln=True)
        self.pdf.cell(0, 8, f"Product Profile: {sample_info.get('product_profile')}", ln=True)
        self.pdf.ln(5)

    def _add_verdict_section(self, analysis_result: AnalysisResult):
        self.pdf.set_font("DejaVu", "B", 12)
        self.pdf.cell(0, 8, f"Final Verdict: {analysis_result.verdict}", ln=True)
        self.pdf.set_font("DejaVu", "", 9)
        if analysis_result.reason_codes:
            self.pdf.multi_cell(0, 5, "Reasons:\n" + "\n".join(f"- {code}" for code in analysis_result.reason_codes))
        else:
            self.pdf.multi_cell(0, 5, "Sample meets all specifications for the selected profile.")
        self.pdf.ln(5)

    def _add_results_table(self, results_dict: dict):
        self.pdf.set_font("DejaVu", "B", 12)
        self.pdf.cell(0, 10, "Detailed Results", ln=True)
        self.pdf.set_font("DejaVu", "", 10)
        for key, val in results_dict.items():
            self.pdf.cell(self.pdf.w * 0.5, 8, f"{key}:", border=1)
            if isinstance(val, dict):
                val_str = ", ".join(f"{k}: {v}%" for k, v in val.items())
            else:
                val_str = str(val if val is not None else "N/A")
            self.pdf.cell(self.pdf.w * 0.3, 8, val_str, border=1, ln=True)

# --- Streamlit UI ---

def initialize_session_state():
    if "history" not in st.session_state: st.session_state.history = pd.DataFrame()
    if "last_run" not in st.session_state: st.session_state.last_run = {}

def build_sidebar(config: AppConfig) -> tuple[SampleData | None, bool]:
    """Creates the sidebar UI for inputs."""
    with st.sidebar:
        st.header("📋 Batch & Sample Info")
        profile = st.selectbox("Select Product Profile", options=list(config.PRODUCT_PROFILES.keys()))
        sample_id = st.text_input("Sample ID", f"sample-{uuid.uuid4().hex[:6]}")
        date = st.date_input("Test Date", datetime.date.today())
        
        st.header("🔬 Test Inputs")
        with st.expander("Particle Size Distribution (PSD)", expanded=True):
            init_mass = st.number_input("Initial Mass (g)", min_value=0.1, value=100.0, step=0.1)
            r02_mass = st.number_input("Retained 0.2–0.4 mm (g)", min_value=0.0, value=30.0, step=0.1)
            r04_mass = st.number_input("Retained >0.4 mm (g)", min_value=0.0, value=20.0, step=0.1)
            fines_pct = st.number_input("Fines (<0.063 mm) (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)

        with st.expander("Equivalent Sand (ES)", expanded=True):
            atb = st.number_input("ATB (NTU)", min_value=0.0, value=50.0, step=0.1)
            atk = st.number_input("ATK (NTU)", min_value=0.1, value=200.0, step=0.1)

        with st.expander("Methylene Blue (MBV)", expanded=True):
            mb_vol = st.number_input("MBV Volume (mL)", min_value=0.0, value=1.0, step=0.1)
            mb_conc = st.number_input("MBV Conc. (mg/mL)", min_value=0.0, value=19.5, step=0.01)
            mb_sample_mass = st.number_input("MBV Sample Mass (g)", min_value=0.1, value=5.0, step=0.1)

        notes = st.text_area("Notes / Observations", "Standard sample.")
        run_button = st.button("🧪 Run Analysis", use_container_width=True)

        inputs = SampleData(profile, sample_id, date, init_mass, r02_mass, r04_mass, fines_pct, atb, atk, mb_vol, mb_conc, mb_sample_mass, notes)
        return inputs, run_button

def build_main_view():
    """Builds the main panel UI for results and history."""
    st.title("Sand Quality Control Dashboard") # Removed Chinese text

    if st.session_state.last_run:
        last_run = st.session_state.last_run
        analysis: AnalysisResult = last_run["analysis_result"]
        
        st.subheader(f"📄 Last Sample: {last_run['sample_info']['sample_id']}")

        col1, col2 = st.columns([3, 1])
        with col1:
            color = analysis.verdict.lower()
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; color: white;">
                <h2 style="text-align: center;">Verdict: {analysis.verdict}</h2>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.image(last_run['qr_image_buffer'], caption="Sample QR Code")

        with st.expander("Verdict Details & Recommendations", expanded=True):
            if analysis.verdict == "GREEN":
                st.success("✅ Sample meets all specifications for the selected profile.")
            elif analysis.verdict == "AMBER":
                st.warning("⚠️ Sample is out of spec but data is valid. Blend may be required.")
                st.markdown("**Reason Codes:**")
                for code in analysis.reason_codes:
                    st.markdown(f"- `{code}`")
                st.info("💡 **Suggestion:** A blend with sand from other silos may be necessary. *[Blend engine logic to be implemented here]*")
            elif analysis.verdict == "RED":
                st.error("⛔️ Data integrity error. Please check your inputs.")
                st.markdown("**Reason Codes:**")
                for code in analysis.reason_codes:
                    st.markdown(f"- `{code}`")
        
        st.json(analysis.results_dict)
        
        st.download_button(
            "⬇️ Download PDF Report",
            data=last_run['pdf_bytes'],
            file_name=f"Report_{last_run['sample_info']['sample_id']}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    if not st.session_state.history.empty:
        st.subheader("📒 Batch History")
        st.dataframe(st.session_state.history)

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    config = AppConfig()
    config.check_font_paths()
    initialize_session_state()
    
    inputs, run_pressed = build_sidebar(config)
    
    if run_pressed and inputs:
        analysis_result = SandAnalysis.perform_full_analysis(inputs, config)
        
        sample_info = {
            "sample_id": inputs.sample_id,
            "date": inputs.date,
            "product_profile": inputs.product_profile,
        }
        
        # --- Dual QR Code Pipeline ---
        qr_str = f"ID:{inputs.sample_id}|Date:{inputs.date.isoformat()}|Profile:{inputs.product_profile}|Verdict:{analysis_result.verdict}"
        qr_img = generate_qr_code(qr_str)
        
        qr_display_buffer = BytesIO()
        qr_img.save(qr_display_buffer, format="PNG")
        qr_display_buffer.seek(0)
        
        pdf_bytes = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                qr_img.save(tmpfile, format="PNG")
                tmp_path = tmpfile.name
            
            pdf_generator = PDFReport(config.FONT_PATH, config.FONT_PATH_BOLD)
            pdf_bytes = pdf_generator.generate(sample_info, analysis_result, tmp_path)
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        st.session_state.last_run = {
            "sample_info": sample_info,
            "analysis_result": analysis_result,
            "pdf_bytes": pdf_bytes,
            "qr_image_buffer": qr_display_buffer,
        }

        history_record = {**sample_info, "Verdict": analysis_result.verdict, **analysis_result.results_dict}
        if 'Clay Content Range (%)' in history_record:
            clay_range = history_record.pop('Clay Content Range (%)')
            history_record.update({f"Clay ({k})": v for k, v in clay_range.items()})

        st.session_state.history = pd.concat([st.session_state.history, pd.DataFrame([history_record])], ignore_index=True)
        
        st.rerun()

    build_main_view()

if __name__ == "__main__":
    main()
