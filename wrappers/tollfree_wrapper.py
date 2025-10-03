# wrappers/tollfree_wrapper.py
import pandas as pd
from agents.tollfree_agent.main import run_tollfree_agent

def run_tollfree_wrapper(df: pd.DataFrame, pattern: str = None) -> pd.DataFrame:
    """
    Wrapper to call the Toll-Free Agent.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Work Phone' column
        pattern (str, optional): Formatting pattern (e.g. "000-000-0000")

    Returns:
        pd.DataFrame: DataFrame with toll-free detection results
    """
    return run_tollfree_agent(df, pattern)
