# wrappers/mailname_wrapper.py
import pandas as pd
import subprocess
import sys
import io
from pathlib import Path
import logging
from tempfile import NamedTemporaryFile

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_mailname_agent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the MailName agent on a DataFrame using in-memory optimizations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to process.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame.
    """
    try:
        # ✅ Save DataFrame to a temporary CSV file
        with NamedTemporaryFile(delete=False, suffix=".csv") as tmp_input:
            df.to_csv(tmp_input.name, index=False)

        tmp_output = Path(tmp_input.name.replace(".csv", "_output.xlsx"))

        # ✅ Path to MailName agent main.py
        mailname_main = Path(__file__).resolve().parents[1] / "agents" / "mailname" / "main.py"
        if not mailname_main.exists():
            raise FileNotFoundError(f"MailName agent main.py not found: {mailname_main}")

        # ✅ Run MailName CLI using same Python interpreter
        subprocess.run(
            [sys.executable, str(mailname_main), "--input", tmp_input.name, "--output", str(tmp_output)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # ✅ Verify output
        if not tmp_output.exists():
            raise FileNotFoundError("MailName agent did not produce the expected output file.")

        # ✅ Load processed output
        processed_df = pd.read_excel(tmp_output, dtype=str)

        # ✅ Optional: Clean up temp files
        Path(tmp_input.name).unlink(missing_ok=True)
        tmp_output.unlink(missing_ok=True)

        logger.info("MailName agent completed successfully with %d rows.", len(processed_df))
        return processed_df

    except subprocess.CalledProcessError as e:
        logger.error("MailName subprocess failed:\n%s", e.stderr)
        raise RuntimeError(f"MailName agent failed: {e.stderr}") from e
    except Exception as e:
        logger.exception("MailName agent crashed: %s", e)
        raise
