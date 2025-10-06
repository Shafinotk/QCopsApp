# wrappers/linkedin_wrapper.py
import io
import sys
import os
import importlib.util
import pandas as pd
import logging
from typing import Optional

# ------------------------------------------
# Logging setup
# ------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------
# Constants
# ------------------------------------------
AGENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'agents', 'linkedin_agent'))


# ------------------------------------------
# Helper: dynamic module loader
# ------------------------------------------
def _load_module_from_path(path: str, module_name: str, pkg_path: Optional[str] = None):
    """Dynamically load a Python module from the given file path."""
    if pkg_path and pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------
# Main LinkedIn Wrapper Function
# ------------------------------------------
def run_linkedin_agent(df: pd.DataFrame, fail_gracefully: bool = True) -> pd.DataFrame:
    """
    Enrich DataFrame with LinkedIn profile URLs using linkedin_agent.file_handler.

    Args:
        df (pd.DataFrame): Input DataFrame containing company/person fields.
        fail_gracefully (bool): Continue pipeline even if agent fails.

    Returns:
        pd.DataFrame: Enriched DataFrame or original DataFrame on failure.
    """
    df = df.copy()

    try:
        fh_path = os.path.join(AGENT_DIR, "file_handler.py")
        if not os.path.exists(fh_path):
            raise FileNotFoundError(f"LinkedIn agent file not found: {fh_path}")

        mod = _load_module_from_path(fh_path, "linkedin_file_handler", pkg_path=AGENT_DIR)
        if not hasattr(mod, "process_csv"):
            raise AttributeError("linkedin_agent.file_handler must define a function `process_csv(bytes_blob)`")

        logger.info("[LinkedIn] Running LinkedIn agent on %d rows...", len(df))

        # ✅ Convert DataFrame to bytes for LinkedIn agent
        csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")

        # ✅ Process the CSV bytes (agent handles its logic)
        processed_bytes = mod.process_csv(csv_bytes)
        if not processed_bytes:
            raise ValueError("LinkedIn agent returned no data.")

        # ✅ Read processed data directly from memory
        out_df = pd.read_csv(io.BytesIO(processed_bytes), encoding="utf-8", dtype=str, keep_default_na=False)

        logger.info("[LinkedIn] Agent completed successfully. Output rows: %d", len(out_df))
        return out_df

    except Exception as e:
        logger.error("[LinkedIn] Agent failed: %s", str(e))
        if fail_gracefully:
            logger.warning("Continuing pipeline despite LinkedIn failure.")
            return df
        else:
            raise


# ------------------------------------------
# Optional AI enhancement hook (future use)
# ------------------------------------------
def enrich_with_ai_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional placeholder for integrating open-source NLP models
    to improve LinkedIn link prediction or name matching.
    Example: Use transformers or spaCy for semantic enrichment.
    """

    # Example (pseudo-code for future use):
    # from transformers import pipeline
    # model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    # df["embedding_vector"] = df["Company Name"].apply(lambda x: model(x)[0])
    # return df

    # Placeholder return for now
    return df

# ------------------------------------------
# Local test
# ------------------------------------------
if __name__ == "__main__":
    test_df = pd.DataFrame([{
        "First Name": "John",
        "Last Name": "Doe",
        "Company Name": "Acme",
        "Title": "CEO",
        "Domain": "acme.com"
    }])

    print(run_linkedin_agent(test_df))
