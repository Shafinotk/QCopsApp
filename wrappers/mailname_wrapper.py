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
    df.to_csv(tmp_input, index=False)

    # Path to the MailName agent main.py
    mailname_main = Path(__file__).resolve().parents[1] / "agents" / "mailname" / "main.py"
    if not mailname_main.exists():
        raise FileNotFoundError(f"MailName agent main.py not found: {mailname_main}")

    # Run MailName CLI
    subprocess.run(
        ["python", str(mailname_main), "--input", str(tmp_input), "--output", str(tmp_output)],
        check=True
    )

    if not tmp_output.exists():
        raise FileNotFoundError("MailName agent did not produce the expected output.")

    # Read processed output
    processed_df = pd.read_excel(tmp_output)

    # Clean up temp files if needed
    tmp_input.unlink(missing_ok=True)
    tmp_output.unlink(missing_ok=True)

    return processed_df
