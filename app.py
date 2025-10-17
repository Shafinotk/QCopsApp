import streamlit as st
import pandas as pd
import tempfile
import io
from pathlib import Path
import chardet
import sys, os
import logging
from openpyxl import Workbook

# ---------------------------
# Ensure project root is on sys.path
# ---------------------------
ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import wrappers
from wrappers.mailname_wrapper import run_mailname_agent
from wrappers.qc_domain_wrapper import run_qc_domain_agent
from wrappers.irvalue_wrapper import run_irvalue_agent
from wrappers.linkedin_wrapper import run_linkedin_agent
from wrappers.tollfree_wrapper import run_tollfree_wrapper
from wrappers.abm_wrapper import run_abm_wrapper
from wrappers.list_checker_wrapper import run_list_checker_wrapper

# Import properization utils
from utils.properization import apply_properization, apply_pobox_coloring

# Import coloring functions
from agents.mailname.qc_checker import apply_mailname_coloring
from agents.qc_domain.qc_agent.io_utils import apply_qc_domain_coloring
from agents.irvalue_phase_4.irvalue_checker import apply_irvalue_coloring
from agents.tollfree_agent.utils import apply_tollfree_coloring
from agents.list_checker_agent.main import apply_list_checker_coloring


# ---------------------------
# Logging setup
# ---------------------------
logger = logging.getLogger("app")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------------------------
# Streamlit UI config
# ---------------------------
st.set_page_config(page_title="Combined Data Processing Agent", layout="wide")
st.title("📊 Combined Data Processing Agent")
st.markdown(
    "Upload your data and run through IRValue, MailName, LinkedIn, QC Domain, ABM Matching, List Checker, and Properization agents."
)

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ---------------------------
# Session persistence
# ---------------------------
if "processed_chunks" not in st.session_state:
    st.session_state["processed_chunks"] = []
if "last_index" not in st.session_state:
    st.session_state["last_index"] = 0
if "partial_file" not in st.session_state:
    st.session_state["partial_file"] = None

# ---------------------------
# CSV reading with encoding detection
# ---------------------------
def read_csv_flexible_bytes(uploaded_bytes) -> pd.DataFrame:
    result = chardet.detect(uploaded_bytes)
    encoding = result["encoding"] if result["encoding"] else "utf-8"
    try:
        df = pd.read_csv(io.BytesIO(uploaded_bytes), encoding=encoding, dtype=str, keep_default_na=False)
    except Exception:
        for enc in ["utf-8", "ISO-8859-1", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(io.BytesIO(uploaded_bytes), encoding=enc, dtype=str, keep_default_na=False)
                break
            except Exception:
                continue
        else:
            raise
    return df

# ---------------------------
# Process single chunk
# ---------------------------
def process_chunk(chunk, run_irvalue, irvalue_fields, run_tollfree, tollfree_pattern,
                  enforce_common_street, run_abm=False, abm_df=None, abm_filename=None, abm_type=None,
                  td_list_enabled=False, run_list_checker=False, competitor_df=None, suppression_df=None, td_df_extra=None):
    try:
        if run_abm and abm_df is not None:
            chunk = run_abm_wrapper(chunk, abm_df, abm_filename, abm_type, td_list_enabled)
        if run_list_checker:
            chunk = run_list_checker_wrapper(chunk, competitor_df, suppression_df, td_df_extra, td_list_enabled)
        if run_irvalue:
            chunk = run_irvalue_agent(chunk, fields=irvalue_fields)
        chunk = run_mailname_agent(chunk)
        if run_tollfree:
            chunk = run_tollfree_wrapper(chunk, tollfree_pattern if tollfree_pattern.strip() else None)
        chunk = run_linkedin_agent(chunk)
        chunk = run_qc_domain_agent(chunk)
        chunk = apply_properization(chunk, enforce_common_street=enforce_common_street)
    except Exception as e:
        logger.exception("Chunk processing failed: %s", e)
        st.warning(f"⚠️ Error processing chunk: {e}")
    return chunk

# ---------------------------
# Main pipeline
# ---------------------------
if uploaded_file:
    tmp_dir = tempfile.mkdtemp()
    input_path = Path(tmp_dir) / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    # Options
    col1, col2 = st.columns(2)
    with col1:
        run_irvalue = st.checkbox("Run IRValue Agent (phase_1)", value=False)
        if run_irvalue:
            irvalue_fields = st.multiselect(
                "Select IRValue fields to fetch",
                options=["employees", "revenue", "industry"],
                default=["employees", "revenue", "industry"],
                help="Choose which fields to enrich from IRValue"
            )
        else:
            irvalue_fields = []
    with col2:
        enforce_common_street = st.checkbox("Enforce most common street per Domain", value=True)
    with st.expander("☎️ Toll-Free Number Options"):
        run_tollfree = st.checkbox("Run Toll-Free Agent", value=False)
        tollfree_pattern = st.text_input("Optional Phone Pattern", placeholder="e.g., 000-000-0000")

    resume = st.checkbox("Resume from last progress", value=False)

    # ---------------------------
    # Unified TD List Mode (Shared)
    # ---------------------------
    td_list_enabled = st.checkbox(
        "Enable TD List Mode",
        value=False,
        help="Use this TD List mode for both ABM and List Checker agents."
    )

    # ---------------------------
    # ABM Matching Agent
    # ---------------------------
    run_abm = st.checkbox("Run ABM Matching Agent", value=False)
    abm_df = None
    abm_filename = None
    abm_type = None

    if run_abm:
        abm_type = st.radio(
            "Select ABM Mode:",
            ["BNZSA QC", "Merit Campaign"],
            horizontal=True,
            help="BNZSA QC marks with ABM filename, Merit Campaign marks as 'ABM'."
        )

        abm_file = st.file_uploader("Upload ABM List (Excel/CSV)", type=["csv", "xlsx"], key="abm")
        if abm_file:
            abm_filename = abm_file.name
            if abm_file.name.lower().endswith(".csv"):
                abm_df = pd.read_csv(abm_file, dtype=str, keep_default_na=False)
            else:
                abm_df = pd.read_excel(abm_file, dtype=str)
            st.success(f"ABM file uploaded: {abm_filename}")

    # ---------------------------
    # List Checker Agent
    # ---------------------------
    run_list_checker = st.checkbox("Run List Checker Agent", value=False)
    competitor_df = suppression_df = td_df_extra = None

    if run_list_checker:
        st.subheader("📋 List Checker Configuration")
        colA, colB, colC = st.columns(3)

        with colA:
            comp_selected = st.checkbox("Competitor List", value=False)
            if comp_selected:
                comp_file = st.file_uploader("Upload Competitor List", type=["csv", "xlsx"], key="comp_list")
                if comp_file:
                    competitor_df = pd.read_excel(comp_file, dtype=str) if comp_file.name.endswith(".xlsx") else pd.read_csv(comp_file, dtype=str)
                    st.success("Competitor list uploaded.")

        with colB:
            sup_selected = st.checkbox("Suppression List", value=False)
            if sup_selected:
                sup_file = st.file_uploader("Upload Suppression List", type=["csv", "xlsx"], key="sup_list")
                if sup_file:
                    suppression_df = pd.read_excel(sup_file, dtype=str) if sup_file.name.endswith(".xlsx") else pd.read_csv(sup_file, dtype=str)
                    st.success("Suppression list uploaded.")

        with colC:
            if td_list_enabled:
                td_file = st.file_uploader("Upload TD List", type=["csv", "xlsx"], key="td_list")
                if td_file:
                    td_df_extra = pd.read_excel(td_file, dtype=str) if td_file.name.endswith(".xlsx") else pd.read_csv(td_file, dtype=str)
                    st.success("TD list uploaded.")

    # ---------------------------
    # Run Pipeline
    # ---------------------------
    if st.button("▶️ Run Pipeline"):
        try:
            # Load data
            if uploaded_file.name.lower().endswith(".csv"):
                raw = uploaded_file.getvalue()
                df = read_csv_flexible_bytes(raw)
            else:
                df = pd.read_excel(input_path, dtype=str)

            df.columns = df.columns.str.strip()
            st.write(f"Rows: {len(df)} | Columns: {list(df.columns)}")

            chunk_size = 20
            progress_bar = st.progress(0)
            status = st.empty()

            partial_path = Path(tmp_dir) / "partial_results.xlsx"
            st.session_state["partial_file"] = partial_path

            if not partial_path.exists():
                wb = Workbook()
                ws = wb.active
                wb.save(partial_path)

            start_index = st.session_state["last_index"] if resume else 0

            for i in range(start_index, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()
                status.write(f"Processing rows {i + 1}–{min(i + chunk_size, len(df))}...")

                processed_chunk = process_chunk(
                    chunk, run_irvalue, irvalue_fields, run_tollfree, tollfree_pattern,
                    enforce_common_street, run_abm, abm_df, abm_filename, abm_type, td_list_enabled,
                    run_list_checker, competitor_df, suppression_df, td_df_extra
                )

                st.session_state["processed_chunks"].append(processed_chunk)

                with pd.ExcelWriter(partial_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                    processed_chunk.to_excel(
                        writer, index=False, header=False,
                        startrow=writer.sheets["Sheet"].max_row
                    )

                st.session_state["last_index"] = i + chunk_size
                progress_bar.progress(min((i + chunk_size) / len(df), 1.0))

            df = pd.concat(st.session_state["processed_chunks"], ignore_index=True)
            final_excel = Path(tmp_dir) / "final_output.xlsx"
            df.to_excel(final_excel, index=False)

            try:
                apply_irvalue_coloring(final_excel)
                apply_mailname_coloring(final_excel)
                apply_qc_domain_coloring(final_excel)
                apply_tollfree_coloring(final_excel, df)
                apply_pobox_coloring(final_excel, df)
                
                if run_list_checker:
                    apply_list_checker_coloring(final_excel)

                
                
                
            except Exception as e:
                st.warning("⚠️ Coloring failed")
                logger.warning("Coloring failed: %s", e)

            st.success("✅ Pipeline completed successfully")
            st.dataframe(df.sample(min(30, len(df))))

            with open(final_excel, "rb") as f:
                st.download_button(
                    label="📥 Download Final Output with Coloring",
                    data=f.read(),
                    file_name="final_output.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error("❌ Error during processing")
            st.exception(e)
            logger.exception("Pipeline failed: %s", e)
