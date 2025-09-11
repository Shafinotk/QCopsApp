# utils/properization.py
import pandas as pd
import re
from collections import Counter

# Direction tokens that should always be uppercase
DIRECTION_TOKENS = {"N", "S", "E", "W", "NE", "NS", "NW", "SN", "SE", "SW", "ES", "EW", "EN", "WE", "WS", "WN"}

# Common street suffixes that should be Title-case (St, Rd, Ave...)
STREET_SUFFIXES = {"ST", "RD", "AVE", "BLVD", "DR", "LN", "CT", "PL", "PKWY", "TER", "CIR", "WAY"}

ZIPCODE_RE = re.compile(r"^\d+$")  # match only digits


def clean_text(value: str) -> str:
    """Properize a text value (remove special chars, fix spacing, proper case)."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    # Remove unwanted characters , . ' : ( ) ;
    value = re.sub(r"[,\.\'\:\(\);]", "", value)
    # Replace multiple spaces or dashes with single space
    value = re.sub(r"[\s\-]+", " ", value)
    # Title case
    return value.title() if value else ""


def _propercase_street_token(token: str) -> str:
    """Properize individual tokens inside a street name."""
    if not token:
        return token
    upper = token.upper()
    if upper in DIRECTION_TOKENS:
        return upper
    if upper in STREET_SUFFIXES:
        return upper.title()  # -> St, Rd, Ave, etc.
    # Handle ordinals (1ST -> 1st)
    m = re.fullmatch(r"(\d+)(ST|ND|RD|TH)", token, flags=re.IGNORECASE)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    return token.title()


def clean_street(value: str) -> str:
    """Special cleaning rules for Street column."""
    if pd.isna(value):
        return ""
    value = str(value).strip()

    # Remove unwanted characters first
    value = re.sub(r"[,\.\'\:\(\);]", "", value)

    # Replace hyphens with space
    value = value.replace("-", " ")

    # Split into tokens and propercase each one
    tokens = [_propercase_street_token(tok) for tok in value.split()]
    return " ".join(tokens)


def _properize_zip(zipcode: str, country: str) -> str:
    """Normalize US ZIP codes. Returns formatted or flagged string."""
    if pd.isna(zipcode):
        return ""
    zipcode = str(zipcode).strip()
    country = str(country).strip().upper()

    if country in ("US", "UNITED STATES"):
        if ZIPCODE_RE.match(zipcode):
            if len(zipcode) == 4:
                return f"`0{zipcode}`"  # add leading zero and wrap with backticks
            elif len(zipcode) == 5:
                return zipcode
            else:
                return f"INVALID:{zipcode}"  # for later highlighting
        else:
            return f"INVALID:{zipcode}"
    return zipcode  # leave as-is for non-US countries


def apply_properization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply properization rules to specific columns in DataFrame."""
    df = df.copy()

    # Normalize text columns
    for col in ["Company Name", "First Name", "Last Name", "Street", "City"]:
        if col in df.columns:
            df[col] = df[col].apply(clean_street if col == "Street" else clean_text)

    # Optional: enforce most common street per domain if Domain column exists
    if "Domain" in df.columns and "Street" in df.columns:
        chosen = {}
        for domain, group in df.groupby("Domain"):
            non_empty = [s for s in group["Street"] if s]
            chosen_val = Counter(non_empty).most_common(1)[0][0] if non_empty else ""
            chosen[domain] = chosen_val
        df["Street"] = df["Domain"].map(lambda d: chosen.get(d, ""))

    # ZIP Code normalization if both Country & Zip Code present
    if "Country" in df.columns and "Zip Code" in df.columns:
        df["Zip Code"] = df.apply(
            lambda row: _properize_zip(row["Zip Code"], row["Country"]),
            axis=1
        )

    return df
