# wrappers/qc_domain_wrapper.py
import pandas as pd
from pathlib import Path
import subprocess

def run_qc_domain_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the QC Domain agent on a DataFrame.

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
    tmp_input = Path("temp_qc_domain_input.csv")
    tmp_output = Path("temp_qc_domain_output.csv")
    df.to_csv(tmp_input, index=False)

    # Path to QC Domain agent main.py
    qc_main = Path(__file__).resolve().parents[1] / "agents" / "qc_domain" / "main.py"
    if not qc_main.exists():
        raise FileNotFoundError(f"QC Domain agent main.py not found: {qc_main}")

    # Run the QC Domain agent via subprocess with proper --input and --output arguments
    subprocess.run(
        ["python", str(qc_main), "--input", str(tmp_input), "--output", str(tmp_output)],
        check=True
    )

    if not tmp_output.exists():
        raise FileNotFoundError("QC Domain agent did not produce the expected output.")

    # Read processed output
    processed_df = pd.read_csv(tmp_output)

    # Optional: clean up temp files
    tmp_input.unlink(missing_ok=True)
    tmp_output.unlink(missing_ok=True)

    return processed_df


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    df = pd.DataFrame({
        "Company Name": ["ACME Inc", "Beta LLC"],
        "Domain": ["acme.com", "beta.com"]
    })
    print(run_qc_domain_agent(df))
