# wrappers/abm_wrapper.py

import pandas as pd
from agents.abm_agent.main import run_abm_agent

def run_abm_wrapper(
    df: pd.DataFrame,
    abm_df: pd.DataFrame,
    abm_filename: str,
    abm_type: str | None = None,
    td_list: bool = False
) -> pd.DataFrame:
    """
    Wrapper to call the ABM Matching Agent.

    Args:
        df (pd.DataFrame): Input dataframe to be processed.
        abm_df (pd.DataFrame): ABM list dataframe.
        abm_filename (str): Name of the uploaded ABM file.
        abm_type (str | None): Type of ABM run (e.g., 'BNZSA QC' or 'Merit Campaign').
        td_list (bool): Whether TD List mode is active (skips using 'Additional Notes').
    """
    return run_abm_agent(df, abm_df, abm_filename, abm_type, td_list)
