import streamlit as st
import pandas as pd
import tempfile
import io
from pathlib import Path
import chardet
import sys, os

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

# Import properization utils
from .utils.properization import apply_properization

# Import coloring functions
from agents.mailname.qc_checker import apply_mailname_coloring
from agents.qc_domain.qc_agent.io_utils import apply_qc_domain_coloring

# ---------------------------
# Streamlit UI config
# ---------------------------
st.set_page_config(page_title="Combined Data Processing Agent", layout="wide")
st.title("📊 Combined Data Processing Agent")
st.markdown(
    "Upload your data and run through MailName, LinkedIn, QC Domain, optional IRValue, and properization."
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
# Main pipeline
# ---------------------------
if uploaded_file:
    tmp_dir = tempfile.mkdtemp()  # Create temp dir
    input_path = Path(tmp_dir) / uploaded_file.name
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File uploaded: {uploaded_file.name}")

    # Optional IRValue checkbox
    run_irvalue = st.checkbox("Run IRValue Agent (phase_4)", value=False)

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
            # Step 1: MailName Agent
            # ---------------------------
            st.write("🔄 Running MailName Agent...")
            df = run_mailname_agent(df)
            st.write(f"After MailName: rows={len(df)}, cols={len(df.columns)}")

            # ---------------------------
            # Step 2: LinkedIn Agent
            # ---------------------------
            st.write("🔄 Running LinkedIn Agent (may take time)...")
            df = run_linkedin_agent(df)
            if "linkedin_link_found" in df.columns:
                count_links = df["linkedin_link_found"].apply(
                    lambda x: bool(str(x).strip())
                ).sum()
                st.write(f"LinkedIn links found: {count_links} / {len(df)}")
            else:
                st.write(
                    "⚠️ LinkedIn agent did not add a linkedin_link_found column."
                )

            # ---------------------------
            # Step 3: QC Domain Agent
            # ---------------------------
            st.write("🔄 Running QC Domain Agent...")
            df = run_qc_domain_agent(df)

            # ---------------------------
            # Step 4: IRValue (optional)
            # ---------------------------
            if run_irvalue:
                st.write("🔄 Running IRValue Agent...")
                df = run_irvalue_agent(df)
            else:
                st.write("⏭️ Skipping IRValue Agent (phase_4)")

            # ---------------------------
            # Step 5: Properization
            # ---------------------------
            st.write("✨ Applying Properization...")
            df = apply_properization(df)

            # ---------------------------
            # Save final output as Excel
            # ---------------------------
            final_excel = Path(tmp_dir) / "final_output.xlsx"
            df.to_excel(final_excel, index=False)

            # ---------------------------
            # Apply colorings
            # ---------------------------
            apply_mailname_coloring(final_excel)  # MailName coloring
            apply_qc_domain_coloring(final_excel)  # QC Domain coloring

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
