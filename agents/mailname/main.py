# agents/mailnameqc/main.py

import os
import argparse
import pandas as pd
import chardet  # encoding detection
import sys
from qc_checker import qc_check  # existing QC logic

# Optional directories (still kept for CLI mode)
UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------- Core DataFrame Logic -------------------- #
def mailname_agent_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply MailName QC to a DataFrame in-memory.
    This wraps the existing file-based qc_check function.
    """
    tmp_input = os.path.join(UPLOAD_DIR, "temp_input.csv")
    tmp_output = os.path.join(RESULT_DIR, "temp_output.xlsx")

    # Write DataFrame to CSV
    df.to_csv(tmp_input, index=False, encoding="utf-8-sig")

    # Run the existing QC checker (file-based)
    qc_check(tmp_input, tmp_output)

    # Load processed QC results
    df_out = pd.read_excel(tmp_output)

    # Cleanup
    if os.path.exists(tmp_input):
        os.remove(tmp_input)
    if os.path.exists(tmp_output):
        os.remove(tmp_output)

    return df_out


# -------------------- CLI Entry Point -------------------- #
def run_cli():
    parser = argparse.ArgumentParser(description="Run MailName QC Checker")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save output XLSX/CSV")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Detect encoding
    with open(args.input, "rb") as f:
        raw_data = f.read(100000)
        detected = chardet.detect(raw_data)
        encoding = detected["encoding"] or "utf-8"

    # Load input DataFrame
    df = pd.read_csv(args.input, encoding=encoding)

    # Save input copy in upload dir
    input_path = os.path.join(UPLOAD_DIR, os.path.basename(args.input))
    df.to_csv(input_path, index=False, encoding="utf-8-sig")

    # Ensure output extension is .xlsx
    base_name, ext = os.path.splitext(args.output)
    output_path = args.output if ext.lower() == ".xlsx" else base_name + ".xlsx"

    # Run QC check
    qc_check(input_path, output_path)

    print(f"[MailName] QC check complete. Output saved to {output_path}")


# -------------------- Script Entry -------------------- #
if __name__ == "__main__":
    run_cli()
