import streamlit as st
import pandas as pd
import tempfile
import io
from pathlib import Path
import chardet
import sys, os
import logging

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
# File upload function
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

def read_csv_flexible_bytes(uploaded_bytes) -> pd.DataFrame:
    """Read CSV with encoding detection and fallbacks."""
    result = chardet.detect(uploaded_bytes)
    encoding = result["encoding"] if result["encoding"] else "utf-8"
    try:
        df = pd.read_csv(
            io.BytesIO(uploaded_bytes),
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
        )
    except Exception:
        for enc in ["utf-8", "ISO-8859-1", "cp1252", "latin1"]:
            try:
                df = pd.read_csv(
                    io.BytesIO(uploaded_bytes),
                    encoding=enc,
                    dtype=str,
                    keep_default_na=False,
                )
                break
            except Exception:
                continue
        else:
            raise
    return df


# ---------------------------
# Session state setup
# ---------------------------
if "final_df" not in st.session_state:
    st.session_state.final_df = None
if "checkpoint_dir" not in st.session_state:
    st.session_state.checkpoint_dir = tempfile.mkdtemp()


# ---------------------------
# Main pipeline
# ---------------------------
if uploaded_file:
    tmp_dir = st.session_state.checkpoint_dir  # persist dir across refresh
    input_path = Path(tmp_dir) / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

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

    if st.button("▶️ Run Pipeline"):
        try:
            # Load input DataFrame
            if uploaded_file.name.lower().endswith(".csv"):
                raw = uploaded_file.getvalue()
                df = read_csv_flexible_bytes(raw)
            else:
                df = pd.read_excel(input_path, dtype=str)

            df.columns = df.columns.str.strip()
            st.write(f"Rows: {len(df)} | Columns: {list(df.columns)}")

            # ---------------------------
            # Step 1: IRValue (optional)
            # ---------------------------
            if run_irvalue:
                st.write("🔄 Running IRValue Agent (first)...")
                try:
                    df = run_irvalue_agent(df, fields=irvalue_fields)
                    st.write(f"IRValue completed: rows={len(df)}")
                    # ✅ Save checkpoint
                    df.to_parquet(Path(tmp_dir) / "checkpoint_irvalue.parquet", index=False)
                    logger.info("IRValue returned %d rows", len(df))
                except Exception as e:
                    st.error("❌ IRValue Agent failed")
                    st.exception(e)
                    logger.exception("IRValue Agent failed: %s", e)
            else:
                st.write("⏭️ Skipping IRValue Agent (phase_1)")

            # ---------------------------
            # Step 2: MailName Agent with CHUNKING
            # ---------------------------
            st.write("🔄 Running MailName Agent in chunks...")
            try:
                chunk_size = 10000
                df_out = []
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    processed = run_mailname_agent(chunk)
                    df_out.append(processed)
                    st.write(f"✅ Processed chunk {i//chunk_size + 1}")
                df = pd.concat(df_out, ignore_index=True)

                # ✅ Save checkpoint
                df.to_parquet(Path(tmp_dir) / "checkpoint_mailname.parquet", index=False)

                st.write(f"After MailName: rows={len(df)}, cols={len(df.columns)}")
                logger.info("MailName completed: rows=%d", len(df))
            except Exception as e:
                st.error("❌ MailName Agent failed")
                st.exception(e)
                logger.exception("MailName Agent failed: %s", e)

            # ---------------------------
            # Step 3: Toll-Free Agent
            # ---------------------------
            if run_tollfree:
                st.write("🔄 Running Toll-Free Agent...")
                try:
                    df = run_tollfree_wrapper(df, tollfree_pattern if tollfree_pattern.strip() else None)
                    df.to_parquet(Path(tmp_dir) / "checkpoint_tollfree.parquet", index=False)  # ✅ checkpoint
                except Exception as e:
                    st.error("❌ Toll-Free Agent failed")
                    st.exception(e)
                    logger.exception("Toll-Free Agent failed: %s", e)
            else:
                st.write("⏭️ Skipping Toll-Free Agent")

            # ---------------------------
            # Step 4: LinkedIn Agent
            # ---------------------------
            st.write("🔄 Running LinkedIn Agent (may take time)...")
            try:
                df = run_linkedin_agent(df)
                df.to_parquet(Path(tmp_dir) / "checkpoint_linkedin.parquet", index=False)  # ✅ checkpoint
                if "linkedin_link_found" in df.columns:
                    count_links = df["linkedin_link_found"].apply(
                        lambda x: bool(str(x).strip())
                    ).sum()
                    st.write(f"LinkedIn links found: {count_links} / {len(df)}")
                    logger.info("LinkedIn links found: %d / %d", count_links, len(df))
                else:
                    st.warning("LinkedIn agent did not add a linkedin_link_found column.")
            except Exception as e:
                st.error("❌ LinkedIn Agent failed")
                st.exception(e)
                logger.exception("LinkedIn Agent failed: %s", e)

            # ---------------------------
            # Step 5: QC Domain Agent
            # ---------------------------
            st.write("🔄 Running QC Domain Agent...")
            try:
                df = run_qc_domain_agent(df)
                df.to_parquet(Path(tmp_dir) / "checkpoint_qc.parquet", index=False)  # ✅ checkpoint
                logger.info("QC Domain Agent completed: rows=%d", len(df))
            except Exception as e:
                st.error("❌ QC Domain Agent failed")
                st.exception(e)
                logger.exception("QC Domain Agent failed: %s", e)

            # ---------------------------
            # Step 6: Properization
            # ---------------------------
            st.write("✨ Applying Properization...")
            try:
                df = apply_properization(df, enforce_common_street=enforce_common_street)
                df.to_parquet(Path(tmp_dir) / "checkpoint_properization.parquet", index=False)  # ✅ checkpoint
                logger.info("Properization applied")
            except Exception as e:
                st.error("❌ Properization failed")
                st.exception(e)
                logger.exception("Properization failed: %s", e)

            # ---------------------------
            # Save final output as Excel
            # ---------------------------
            final_excel = Path(tmp_dir) / "final_output.xlsx"
            df.to_excel(final_excel, index=False)

            # ---------------------------
            # Apply colorings
            # ---------------------------
            try:
                apply_irvalue_coloring(final_excel)
                apply_mailname_coloring(final_excel)
                apply_qc_domain_coloring(final_excel)
                apply_tollfree_coloring(final_excel, df)
                apply_pobox_coloring(final_excel, df)
            except Exception as e:
                st.warning("⚠️ Coloring failed")
                st.exception(e)
                logger.warning("Coloring failed: %s", e)

            # ✅ Save to session state
            st.session_state.final_df = df

            st.success("✅ Pipeline completed successfully")
            st.dataframe(df.head(50))

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
# Restore state on refresh
# ---------------------------
if st.session_state.final_df is not None:
    st.write("📌 Restored from session state / checkpoint")
    st.dataframe(st.session_state.final_df.head(20))
