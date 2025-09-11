# wrappers/mailname_wrapper.py
import subprocess
from pathlib import Path
import pandas as pd
import os

def run_mailname_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the MailName agent on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    # Save df to a temporary CSV
    tmp_input = Path("temp_mailname_input.csv")
    tmp_output = Path("temp_mailname_output.xlsx")
    df.to_csv(tmp_input, index=False, encoding="utf-8-sig")

    # Path to the MailName agent main.py (corrected folder: mailnameqc)
    mailname_main = Path(__file__).resolve().parents[1] / "agents" / "mailnameqc" / "main.py"
    if not mailname_main.exists():
        raise FileNotFoundError(f"MailName agent main.py not found: {mailname_main}")

    # Run MailName CLI (capture output so errors are visible)
    cmd = ["python", str(mailname_main), "--input", str(tmp_input), "--output", str(tmp_output)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(mailname_main.parent))

    if result.returncode != 0:
        raise RuntimeError(
            "MailName agent failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if result.stdout:
        print("[mailname stdout]", result.stdout)

    # Ensure output exists
    if not tmp_output.exists():
        raise FileNotFoundError("MailName agent did not produce the expected output.")

    # Read processed output
    processed_df = pd.read_excel(tmp_output)

    # Clean up temp files
    tmp_input.unlink(missing_ok=True)
    tmp_output.unlink(missing_ok=True)

    return processed_df
