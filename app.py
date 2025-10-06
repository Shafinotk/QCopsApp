import streamlit as st
import pandas as pd
import tempfile
import io
from pathlib import Path
import chardet
import sys, os
import logging
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

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

# Import properization utils
from utils.properization import apply_properization, apply_pobox_coloring

# Import coloring functions
from agents.mailname.qc_checker import apply_mailname_coloring
from agents.qc_domain.qc_agent.io_utils import apply_qc_domain_coloring
from agents.irvalue_phase_4.irvalue_checker import apply_irvalue_coloring
from agents.tollfree_agent.utils import apply_tollfree_coloring

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
    "Upload your data and run through IRValue (optional), MailName, LinkedIn, QC Domain, and properization."
)

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

# ---------------------------
# Session persistence setup
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
def process_chunk(chunk, run_irvalue, irvalue_fields, run_tollfree, tollfree_pattern, enforce_common_street):
    try:
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

    if st.button("▶️ Run Pipeline"):
        try:
            # Load DataFrame
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

            # Create Excel workbook if doesn't exist
            if not partial_path.exists():
                wb = Workbook()
                ws = wb.active
                wb.save(partial_path)

            start_index = st.session_state["last_index"] if resume else 0

            for i in range(start_index, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()
                status.write(f"Processing rows {i + 1}–{min(i + chunk_size, len(df))}...")

                processed_chunk = process_chunk(
                    chunk, run_irvalue, irvalue_fields, run_tollfree, tollfree_pattern, enforce_common_street
                )

                # Save processed chunk in session
                st.session_state["processed_chunks"].append(processed_chunk)

                # Append chunk to Excel (incremental save)
                with pd.ExcelWriter(partial_path, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
                    processed_chunk.to_excel(
                        writer,
                        index=False,
                        header=False,
                        startrow=writer.sheets["Sheet"].max_row
                    )

                st.session_state["last_index"] = i + chunk_size
                progress_bar.progress(min((i + chunk_size) / len(df), 1.0))

                # Checkpoint every 100 rows
                if i % 100 == 0:
                    checkpoint_path = Path(tmp_dir) / f"checkpoint_{i}.xlsx"
                    pd.concat(st.session_state["processed_chunks"]).to_excel(checkpoint_path, index=False)
                    logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Combine all processed chunks
            df = pd.concat(st.session_state["processed_chunks"], ignore_index=True)

            final_excel = Path(tmp_dir) / "final_output.xlsx"
            df.to_excel(final_excel, index=False)

            # Apply coloring
            try:
                apply_irvalue_coloring(final_excel)
                apply_mailname_coloring(final_excel)
                apply_qc_domain_coloring(final_excel)
                apply_tollfree_coloring(final_excel, df)
                apply_pobox_coloring(final_excel, df)
            except Exception as e:
                st.warning("⚠️ Coloring failed")
                logger.warning("Coloring failed: %s", e)

            st.success("✅ Pipeline completed successfully")
            st.dataframe(df.sample(min(30, len(df))))  # Show limited rows

            # Download
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
