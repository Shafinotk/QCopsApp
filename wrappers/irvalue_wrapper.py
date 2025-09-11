# wrappers/irvalue_wrapper.py
import pandas as pd
from pathlib import Path
import subprocess

def run_irvalue_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the IRValue agent on a DataFrame via its CLI.
    Saves the DataFrame to a temporary CSV and reads back the enriched data.
    """
    tmp_input = Path("temp_irvalue_input.csv")
    tmp_output = Path("temp_irvalue_output.csv")

    df.to_csv(tmp_input, index=False, encoding="utf-8")

    # Path to IRValue main.py
    ir_main = Path(__file__).resolve().parents[1] / "agents" / "irvalue_phase_4" / "main.py"
    if not ir_main.exists():
        raise FileNotFoundError(f"IRValue agent main.py not found: {ir_main}")

    # Run IRValue CLI with input/output arguments
    subprocess.run(
        ["python", str(ir_main), "--input", str(tmp_input), "--output", str(tmp_output)],
        check=True
    )

    if not tmp_output.exists():
        raise FileNotFoundError("IRValue agent did not produce the expected output.")

    # Read processed output
    processed_df = pd.read_csv(tmp_output, dtype=str, keep_default_na=False)

    # Cleanup temp files
    tmp_input.unlink(missing_ok=True)
    tmp_output.unlink(missing_ok=True)

    return processed_df


if __name__ == "__main__":
    # Quick test
    df = pd.DataFrame({
        "Company Name": ["Acme Inc", "Beta LLC"],
        "Domain": ["acme.com", "beta.com"],
        "Country": ["USA", "USA"]
    })
    print(run_irvalue_agent(df))
