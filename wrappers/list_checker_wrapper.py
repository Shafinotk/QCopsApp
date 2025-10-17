import pandas as pd
from agents.list_checker_agent.main import run_list_checker_agent

def run_list_checker_wrapper(
    df: pd.DataFrame,
    competitor_df: pd.DataFrame | None = None,
    suppression_df: pd.DataFrame | None = None,
    td_df: pd.DataFrame | None = None,
    td_enabled: bool = False
) -> pd.DataFrame:
    """
    Wrapper for the List Checker Agent.

    This function calls the main agent and returns a processed DataFrame with:
    - Boolean columns: 'Competitor List', 'Suppression List'
    - Optional reason columns: 'Competitor_Reason', 'Suppression_Reason' 
      (added only if the corresponding list is provided)
    - 'TD Status Message' column if TD list mode is enabled
    """
    result_df = run_list_checker_agent(
        df=df,
        competitor_df=competitor_df,
        suppression_df=suppression_df,
        td_df=td_df,
        td_enabled=td_enabled
    )

    return result_df
