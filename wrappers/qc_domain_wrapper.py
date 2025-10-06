# wrappers/qc_domain_wrapper.py
import pandas as pd
from pathlib import Path
import subprocess
import sys
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

@st.cache_data(show_spinner=False)
def run_qc_domain_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the QC Domain agent on a DataFrame (cached).
    If FAISS semantic index is available in session, enrich domain names semantically.
    """
    # Save df to a temporary CSV
    tmp_input = Path("temp_qc_domain_input.csv")
    tmp_output = Path("temp_qc_domain_output.csv")
    df.to_csv(tmp_input, index=False)

    qc_main = Path(__file__).resolve().parents[1] / "agents" / "qc_domain" / "main.py"
    if not qc_main.exists():
        raise FileNotFoundError(f"QC Domain agent main.py not found: {qc_main}")

    # Run the QC Domain agent subprocess
    subprocess.run(
        [sys.executable, str(qc_main), "--input", str(tmp_input), "--output", str(tmp_output)],
        check=True
    )

    if not tmp_output.exists():
        raise FileNotFoundError("QC Domain agent did not produce expected output.")

    processed_df = pd.read_csv(tmp_output, dtype=str)

    # Semantic domain enrichment (optional)
    if "Domain" in processed_df.columns and "faiss_index" in st.session_state:
        index, model, known_domains = st.session_state["faiss_index"]
        new_domains = processed_df["Domain"].fillna("").tolist()
        new_embs = model.encode(new_domains, show_progress_bar=False)
        new_embs = np.array(new_embs).astype("float32")
        faiss.normalize_L2(new_embs)
        D, I = index.search(new_embs, k=1)
        processed_df["Closest_Known_Domain"] = [known_domains[i[0]] if len(i) else "" for i in I]

    # Clean up temp files
    tmp_input.unlink(missing_ok=True)
    tmp_output.unlink(missing_ok=True)
    return processed_df
