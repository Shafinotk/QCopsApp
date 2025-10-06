# wrappers/irvalue_wrapper.py
import pandas as pd
import traceback
import inspect
import sys
import os
import asyncio
import logging
from typing import Optional, List

# ------------------------------------------
# Logging setup
# ------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------------------
# Import IRValue logic
# ------------------------------------------
try:
    from agents.irvalue_phase_4.main import irvalue_logic
except Exception:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from agents.irvalue_phase_4.main import irvalue_logic


# ------------------------------------------
# Main Function
# ------------------------------------------
def run_irvalue_agent(
    df: pd.DataFrame,
    debug: bool = False,
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Run the IRValue agent in-process (no subprocess) for efficiency.
    Adds intelligent error handling and async support.

    Args:
        df (pd.DataFrame): Input dataframe with at least Domain, Country, Company Name
        debug (bool): Enable debug logs.
        fields (list[str] | None): Fields to fetch (['employees', 'revenue', 'industry'])

    Returns:
        pd.DataFrame: Enriched DataFrame or placeholder DataFrame on failure.
    """
    df = df.copy()

    # Ensure required columns exist
    for col in ["Domain", "Country", "Company Name"]:
        if col not in df.columns:
            df[col] = ""

    try:
        # -----------------------------
        # Detect function signature
        # -----------------------------
        sig = inspect.signature(irvalue_logic)
        accepts_fields = "fields" in sig.parameters

        # -----------------------------
        # Async support
        # -----------------------------
        if asyncio.iscoroutinefunction(irvalue_logic):
            enriched = asyncio.run(
                irvalue_logic(df, debug=debug, fields=fields) if accepts_fields
                else irvalue_logic(df, debug=debug)
            )
        else:
            enriched = (
                irvalue_logic(df, debug=debug, fields=fields)
                if accepts_fields else irvalue_logic(df, debug=debug)
            )

        # -----------------------------
        # Validation
        # -----------------------------
        if not isinstance(enriched, pd.DataFrame):
            raise RuntimeError("IRValue agent returned a non-DataFrame result")

        # ✅ Optional: add post-processing hooks here (AI enrichment)
        # Example: add semantic checks, ML anomaly detection, etc.
        # enriched = enrich_with_ai_model(enriched)

        logger.info("IRValue agent completed successfully with %d rows", len(enriched))
        return enriched

    except Exception as e:
        logger.error("IRValue agent failed: %s", str(e))
        traceback.print_exc()

        # -----------------------------
        # Fallback DataFrame structure
        # -----------------------------
        placeholder_cols = [
            "discovered_employees_raw",
            "discovered_revenue_raw",
            "discovered_industry",
            "flagged_rpe",
            "discovered_employees",
            "discovered_revenue",
        ]
        for c in placeholder_cols:
            if c not in df.columns:
                df[c] = ""

        logger.warning("Returning fallback DataFrame due to IRValue failure.")
        return df


# ------------------------------------------
# Optional AI enrichment hook (future use)
# ------------------------------------------
def enrich_with_ai_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional placeholder for integrating an open-source model to refine
    or validate IRValue outputs — ready for future extension.
    """
    # Example (pseudo-code):
    # from transformers import pipeline
    # model = pipeline("text-classification", model="distilbert-base-uncased")
    # df["ai_validated"] = df["Company Name"].apply(lambda x: model(x)[0]["label"])
    return df


# ------------------------------------------
# Local test
# ------------------------------------------
if __name__ == "__main__":
    test_df = pd.DataFrame({
        "Company Name": ["Example Inc", "Foo LLC"],
        "Domain": ["example.com", "foo.com"],
        "Country": ["USA", "USA"]
    })

    print(run_irvalue_agent(test_df, debug=True, fields=["employees", "revenue"]).head())
