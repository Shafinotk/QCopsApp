# agents/qc_domain_agent/main.py
import os
import pandas as pd
import asyncio
from dotenv import load_dotenv
from qc_agent.pipeline import process_dataframe_async, merge_with_original
from qc_agent.io_utils import guess_columns, save_outputs
import chardet  # added for encoding detection

load_dotenv()

def qc_domain_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply QC Domain checks to a DataFrame and return the enriched DataFrame.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    company_col, domain_col = guess_columns(df)

    if not company_col and "Company Name" in df.columns:
        company_col = "Company Name"
    if not domain_col and "Domain" in df.columns:
        domain_col = "Domain"

    if not company_col:
        raise ValueError("Could not detect the company column. Make sure your DataFrame has a company column.")
    if not domain_col:
        print("Warning: Could not detect domain column; proceeding without it.")

    qc = asyncio.run(
        process_dataframe_async(df, company_col, domain_col, provider="ddg", limit=5, timeout=12)
    )

    merged = merge_with_original(df, qc)
    return merged


def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run QC Domain Agent")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save output CSV/XLSX")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw_data = f.read(100000)
        detected = chardet.detect(raw_data)
        encoding = detected["encoding"] or "utf-8"

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding=encoding)
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    result_df = qc_domain_logic(df)

    # Save CSV/XLSX (no coloring here)
    save_outputs(result_df, args.output, args.output.replace(".csv", ".xlsx"))
    print(f"QC Domain processing complete. Output saved to {args.output}")


if __name__ == "__main__":
    run_cli()
