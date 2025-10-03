# agents/tollfree_agent/main.py
import pandas as pd
import re
from agents.tollfree_agent.utils import (
    normalize_number,
    format_number,
    is_toll_free,
    COUNTRY_ALIASES,
)

def run_tollfree_agent(df: pd.DataFrame, pattern: str = None) -> pd.DataFrame:
    """
    Update 'Work Phone' in-place, detect toll-free/invalid numbers,
    and create:
      - 'WorkPhone_Reason' (tollfree / other / None)
      - 'WorkPhone_ColorFlag' (True if needs coloring)
    """
    if "Work Phone" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Work Phone' column")

    if "Country" not in df.columns:
        raise ValueError("Input DataFrame must have a 'Country' column")

    df = df.copy()
    reasons = []

    for idx, phone in df["Work Phone"].items():
        number = normalize_number(str(phone))
        country_val = str(df.at[idx, "Country"]).strip()

        # Normalize country
        country_key = COUNTRY_ALIASES.get(country_val.upper(), country_val.upper())

        # Optional length validation (can still keep US/IN rules)
        is_invalid_length = False
        if country_key in ("US", "CA"):
            if number.startswith("1") and len(number) > 10:
                number = number[1:]
            if len(number) < 10:
                is_invalid_length = True
        elif country_key == "IN":
            if number.startswith("91") and len(number[2:]) < 10:
                is_invalid_length = True

        # Pass country_val to format_number
        formatted = format_number(number, country=country_val, pattern=pattern)

        # --- Checks ---
        toll_free = is_toll_free(number, country_key)
        has_zeros = bool(re.search(r"0{6,}", number))

        # --- Assign reason ---
        if toll_free:
            reasons.append("tollfree")
        elif has_zeros or is_invalid_length:
            reasons.append("other")
        else:
            reasons.append(None)

        # Update Work Phone column in-place
        df.at[idx, "Work Phone"] = formatted

    # Add flag + reason columns
    df["WorkPhone_Reason"] = reasons
    df["WorkPhone_ColorFlag"] = [r is not None for r in reasons]

    return df
