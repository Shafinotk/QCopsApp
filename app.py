# app.py (REPLACE your existing file with this content)
import streamlit as st
import pandas as pd
import tempfile
import io
from pathlib import Path
import chardet
import sys, os
import logging
import time
import json

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
# Initialize session state
# ---------------------------
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None
if "tmp_dir" not in st.session_state:
    st.session_state["tmp_dir"] = None

# ---------------------------
# File upload function
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

def read_csv_flexible_bytes(uploaded_bytes) -> pd.DataFrame:
    """Read CSV with encoding detection and fallbacks."""
    result = chardet.detect(uploaded_bytes)
    encoding = result["encoding"] if result and result.get("encoding") else "utf-8"
    tried = []
    for enc in [encoding, "utf-8", "ISO-8859-1", "cp1252", "latin1"]:
        if not enc or enc in tried:
            continue
        tried.append(enc)
        try:
            df = pd.read_csv(
                io.BytesIO(uploaded_bytes),
                encoding=enc,
                dtype=str,
                keep_default_na=False,
            )
            return df
        except Exception as e:
            logger.debug("Encoding %s failed: %s", enc, e)
            continue
    # fallback: try pandas detect separators / engine
    try:
        df = pd.read_csv(io.BytesIO(uploaded_bytes), dtype=str, keep_default_na=False, engine="python")
        return df
    except Exception as e:
        logger.exception("All CSV decoding attempts failed")
        raise

# ---------------------------
# Helper: process one chunk
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
# Utility: checkpointing (save partial results)
# ---------------------------
def checkpoint_chunk(df_chunk, tmp_dir, idx):
    path = Path(tmp_dir) / f"chunk_{idx}.parquet"
    df_chunk.to_parquet(path, index=False)
    return str(path)

def load_checkpoint_files(tmp_dir):
    p = Path(tmp_dir)
    files = sorted(p.glob("chunk_*.parquet"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception:
            logger.exception("Failed loading checkpoint %s", f)
    return dfs

# ---------------------------
# Main pipeline UI & logic
# ---------------------------
if uploaded_file:
    # Save uploaded file into session (persist across reruns)
    if st.session_state.uploaded_file_name != uploaded_file.name:
        tmp_dir = tempfile.mkdtemp()
        st.session_state.tmp_dir = tmp_dir
        st.session_state.uploaded_file_name = uploaded_file.name
        with open(Path(tmp_dir) / uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded: {uploaded_file.name}")
        st.session_state.df = None  # reset any previously processed dataframe

    # Optional checkboxes BEFORE pipeline starts
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
        enforce_common_street = st.checkbox(
            "Enforce most common street per Domain",
            value=True,
            help="If checked, all rows with the same domain will get the most common street value."
        )
    with st.expander("☎️ Toll-Free Number Options"):
        run_tollfree = st.checkbox("Run Toll-Free Agent", value=False)
        tollfree_pattern = st.text_input(
            "Optional Phone Pattern",
            placeholder="e.g., 000-000-0000 or (+00) 000000000",
            help="If left empty, default formatting rules are applied.",
        )

    # Run pipeline
    if st.button("▶️ Run Pipeline"):
        try:
            # Load input DataFrame (maybe from session state)
            if st.session_state.df is None:
                if uploaded_file.name.lower().endswith(".csv"):
                    raw = uploaded_file.getvalue()
                    df = read_csv_flexible_bytes(raw)
                else:
                    df = pd.read_excel(Path(st.session_state.tmp_dir) / uploaded_file.name, dtype=str)
                df.columns = df.columns.str.strip()
                st.session_state.df = df
            else:
                df = st.session_state.df

            st.write(f"Rows: {len(df)} | Columns: {list(df.columns)}")

            # ---------------------------
            # Chunked processing with progress bar + checkpointing
            # ---------------------------
            chunk_size = 20
            df_out_paths = []
            progress_bar = st.progress(0)
            status = st.empty()
            total = len(df)
            chunks = list(range(0, total, chunk_size))
            last_idx = 0

            for idx, i in enumerate(chunks):
                start = i
                end = min(i + chunk_size, total)
                status.write(f"Processing rows {start+1}–{end} (chunk {idx+1}/{len(chunks)}) ...")
                chunk = df.iloc[start:end].copy()

                # Process chunk
                processed_chunk = process_chunk(
                    chunk,
                    run_irvalue,
                    irvalue_fields,
                    run_tollfree,
                    tollfree_pattern,
                    enforce_common_street
                )

                # checkpoint to disk (parquet)
                path = checkpoint_chunk(processed_chunk, st.session_state.tmp_dir, idx)
                df_out_paths.append(path)

                last_idx = end
                progress_bar.progress(min(1.0, end / max(total, 1)))

            # assemble final df from checkpoints
            df_parts = load_checkpoint_files(st.session_state.tmp_dir)
            if df_parts:
                df_final = pd.concat(df_parts, ignore_index=True)
            else:
                df_final = pd.DataFrame()

            st.session_state.df = df_final

            # ---------------------------
            # Save final output as Excel
            # ---------------------------
            final_excel = Path(st.session_state.tmp_dir) / "final_output.xlsx"
            df_final.to_excel(final_excel, index=False)

            # ---------------------------
            # Apply colorings
            # ---------------------------
            try:
                apply_irvalue_coloring(final_excel)
                apply_mailname_coloring(final_excel)
                apply_qc_domain_coloring(final_excel)
                apply_tollfree_coloring(final_excel, df_final)
                apply_pobox_coloring(final_excel, df_final)
            except Exception as e:
                st.warning("⚠️ Coloring failed (non-fatal). Check logs.")
                logger.exception("Coloring failed: %s", e)

            st.success("✅ Pipeline completed successfully")
            st.dataframe(df_final.head(50))

            # ---------------------------
            # Download button for Excel
            # ---------------------------
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

# ---------------------------
# Helpful debugging / housekeeping
# ---------------------------
with st.expander("Debug / Session Info"):
    st.write("Uploaded file:", st.session_state.uploaded_file_name)
    st.write("Tmp dir:", st.session_state.tmp_dir)
    st.write("DF shape:", None if st.session_state.df is None else st.session_state.df.shape)
