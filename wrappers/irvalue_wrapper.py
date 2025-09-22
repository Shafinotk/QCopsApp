# wrappers/irvalue_wrapper.py
import pandas as pd
from pathlib import Path
import subprocess
import sys
import tempfile

def run_irvalue_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the IRValue agent on a DataFrame via its CLI.
    Saves the DataFrame to a temporary CSV and reads back the enriched data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_input = Path(tmpdir) / "input.csv"
        tmp_output = Path(tmpdir) / "output.csv"

        # Save input file
        df.to_csv(tmp_input, index=False, encoding="utf-8")

        # Path to IRValue main.py
        ir_main = Path(__file__).resolve().parents[1] / "agents" / "irvalue_phase_4" / "main.py"
        if not ir_main.exists():
            raise FileNotFoundError(f"IRValue agent main.py not found: {ir_main}")

        # Run subprocess and capture logs
        result = subprocess.run(
            [sys.executable, str(ir_main), "--input", str(tmp_input), "--output", str(tmp_output)],
            capture_output=True,
            text=True
        )

        # Debug logs
        print("=== IRValue STDOUT ===")
        print(result.stdout)
        print("=== IRValue STDERR ===")
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(f"IRValue agent failed with code {result.returncode}")

        if not tmp_output.exists():
            raise FileNotFoundError("IRValue agent did not produce the expected output.")

        # Read processed output
        processed_df = pd.read_csv(tmp_output, dtype=str, keep_default_na=False)

    return processed_df


if __name__ == "__main__":
    df = pd.DataFrame({
        "Company Name": ["Acme Inc", "Beta LLC"],
        "Domain": ["acme.com", "beta.com"],
        "Country": ["USA", "USA"]
    })
    print(run_irvalue_agent(df))
