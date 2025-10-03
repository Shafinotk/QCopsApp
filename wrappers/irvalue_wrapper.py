# wrappers/irvalue_wrapper.py
import pandas as pd
from pathlib import Path
import tempfile
import traceback
import sys
import os
import inspect

# Import the IRValue logic directly (runs in-process instead of subprocess)
try:
    from agents.irvalue_phase_4.main import irvalue_logic
except Exception:
    # Fallback if path issues — try adjusting sys.path
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from agents.irvalue_phase_4.main import irvalue_logic


def run_irvalue_agent(
    df: pd.DataFrame,
    debug: bool = False,
    fields: list[str] | None = None
) -> pd.DataFrame:
    """
    Run the IRValue agent in-process on the given DataFrame.
    This avoids fragile subprocess invocations on cloud platforms.

    Args:
        df (pd.DataFrame): Input dataframe with at least Domain, Country, Company Name
        debug (bool): Enable debug logging
        fields (list[str] | None): Which fields to fetch. Options:
            ["employees", "revenue", "industry"]
            Default = None (fetch all)

    Returns:
        pd.DataFrame: Enriched DataFrame; on failure, returns the original df with
        added empty columns so downstream code doesn't break.
    """
    # Defensive copy
    df = df.copy()

    # Ensure required columns exist
    for col in ["Domain", "Country", "Company Name"]:
        if col not in df.columns:
            df[col] = ""

    try:
        import asyncio

        # check if irvalue_logic supports "fields"
        sig = inspect.signature(irvalue_logic)
        accepts_fields = "fields" in sig.parameters

        if asyncio.iscoroutinefunction(irvalue_logic):
            if accepts_fields:
                enriched = asyncio.run(irvalue_logic(df, debug=debug, fields=fields))
            else:
                enriched = asyncio.run(irvalue_logic(df, debug=debug))
        else:
            if accepts_fields:
                enriched = irvalue_logic(df, debug=debug, fields=fields)
            else:
                enriched = irvalue_logic(df, debug=debug)

        # ensure we return a DataFrame
        if not isinstance(enriched, pd.DataFrame):
            raise RuntimeError("IRValue returned non-DataFrame result")
        return enriched

    except Exception:
        # Log the traceback to stdout so Render/Streamlit logs show it
        print("ERROR running IRValue agent:", file=sys.stderr)
        traceback.print_exc()
        print(
            "IRValue failed — returning original DataFrame with placeholder columns",
            file=sys.stderr,
        )

        # Add the discovered columns (empty) so downstream code won't break
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

        return df


# quick local test if run directly
if __name__ == "__main__":
    df = pd.DataFrame({
        "Company Name": ["Example Inc", "Foo LLC"],
        "Domain": ["example.com", "foo.com"],
        "Country": ["USA", "USA"]
    })

    # Example: only fetch employees (only works if irvalue_logic supports fields)
    print(run_irvalue_agent(df, debug=True, fields=["employees"]).head())

    # Example: fetch all (default)
    print(run_irvalue_agent(df, debug=True).head())
