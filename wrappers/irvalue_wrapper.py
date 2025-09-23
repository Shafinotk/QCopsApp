# wrappers/irvalue_wrapper.py
import pandas as pd
from pathlib import Path
import tempfile
import traceback
import sys
import os

# Import the IRValue logic directly (runs in-process instead of subprocess)
# This assumes your irvalue module exposes irvalue_logic(df, debug=False)
try:
    from agents.irvalue_phase_4.main import irvalue_logic
except Exception:
    # Fallback if path issues — try adjusting sys.path
    repo_root = os.path.dirname(os.path.dirname(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from agents.irvalue_phase_4.main import irvalue_logic

def run_irvalue_agent(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Run the IRValue agent in-process on the given DataFrame.
    This avoids fragile subprocess invocations on cloud platforms.
    Returns the enriched DataFrame; on failure, returns the original df with added empty columns.
    """
    # Defensive copy
    df = df.copy()

    # Ensure required columns exist
    for col in ["Domain", "Country", "Company Name"]:
        if col not in df.columns:
            df[col] = ""

    try:
        # If irvalue_logic is async, run it via asyncio.run
        import asyncio
        # detect if irvalue_logic is coroutine function
        if asyncio.iscoroutinefunction(irvalue_logic):
            enriched = asyncio.run(irvalue_logic(df, debug=debug))
        else:
            # irvalue_logic may be sync and accept df
            enriched = irvalue_logic(df, debug=debug)

        # ensure we return a DataFrame
        if not isinstance(enriched, pd.DataFrame):
            raise RuntimeError("IRValue returned non-DataFrame result")
        return enriched

    except Exception as exc:
        # Log the traceback to stdout so Render/Streamlit logs show it
        print("ERROR running IRValue agent:", file=sys.stderr)
        traceback.print_exc()
        print("IRValue failed — returning original DataFrame with placeholder columns", file=sys.stderr)

        # Add the discovered columns (empty) so downstream code won't break
        placeholder_cols = [
            "discovered_employees_raw",
            "discovered_revenue_raw",
            "discovered_industry",
            "flagged_rpe",
            "discovered_employees",
            "discovered_revenue"
        ]
        for c in placeholder_cols:
            if c not in df.columns:
                df[c] = ""

        return df


# quick local test if run directly
if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({
        "Company Name": ["Example Inc", "Foo LLC"],
        "Domain": ["example.com", "foo.com"],
        "Country": ["USA", "USA"]
    })
    print(run_irvalue_agent(df, debug=True).head())
