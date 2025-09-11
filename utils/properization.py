# utils/properization.py
import pandas as pd
import re
from collections import Counter

# Direction tokens that should always be uppercase
DIRECTION_TOKENS = {
    "N", "S", "E", "W",
    "NE", "NS", "NW", "SN", "SE", "SW",
    "ES", "EW", "EN", "WE", "WS", "WN"
}

# Common street suffixes
STREET_TYPE_MAP = {
    "st": "St", "rd": "Rd", "ave": "Ave", "blvd": "Blvd",
    "ln": "Ln", "dr": "Dr", "hwy": "Hwy", "pkwy": "Pkwy",
    "cir": "Cir", "ct": "Ct", "pl": "Pl", "ter": "Ter", "way": "Way"
}

PUNCT_TO_REMOVE = r"[,\.\'\:\(\);]"
ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)", flags=re.IGNORECASE)
ZIPCODE_RE = re.compile(r"^\d+$")  # match only digits


def _canonical_col(df, target_name: str):
    """Find a column in df that matches target_name ignoring case/spacing/punctuation."""
    tn = re.sub(r"\s+", "", target_name).lower()
    for c in df.columns:
        cc = re.sub(r"\s+", "", c).lower()
        if cc == tn:
            return c
    return None


def _clean_basic_text(value: str) -> str:
    """Remove punctuation, clean spacing, return stripped string."""
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(PUNCT_TO_REMOVE, "", value)
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_text(value: str) -> str:
    """General clean + proper case."""
    value = _clean_basic_text(value)
    return value.title() if value else ""


def _propercase_street_token(token: str) -> str:
    """Properize individual tokens inside a street name."""
    if not token:
        return token
    upper = token.upper()

    # Direction tokens
    if upper in DIRECTION_TOKENS:
        return upper

    # Street suffix normalization
    if upper.lower() in STREET_TYPE_MAP:
        return STREET_TYPE_MAP[upper.lower()]

    # Ordinals (1st, 2nd, etc.)
    m = ORDINAL_RE.fullmatch(token)
    if m:
        num, suf = m.groups()
        return f"{num}{suf.lower()}"

    return token.title()


def clean_street(value: str) -> str:
    """Special cleaning rules for Street column."""
    value = _clean_basic_text(value)
    if not value:
        return value
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
                return f"INVALID:{zipcode}"  # mark invalid
        else:
            return f"INVALID:{zipcode}"
    return zipcode  # leave as-is for non-US countries


def apply_properization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply properization rules to DataFrame."""
    df = df.copy()

    # Map canonical column names (flexible matching)
    mapping = {}
    for target in ["Company Name", "First Name", "Last Name", "Street", "City", "Domain", "Country", "Zip Code"]:
        found = _canonical_col(df, target)
        if found:
            mapping[target] = found

    # Clean and title-case Company/Name/City
    for col in ["Company Name", "First Name", "Last Name", "City"]:
        if col in mapping:
            c = mapping[col]
            df[c] = df[c].fillna("").astype(str).apply(clean_text)

    # Street cleaning
    if "Street" in mapping:
        c = mapping["Street"]
        df[c] = df[c].fillna("").astype(str).apply(clean_street)

    # ZIP code normalization
    if "Zip Code" in mapping and "Country" in mapping:
        zc = mapping["Zip Code"]
        cc = mapping["Country"]
        df[zc] = df.apply(lambda row: _properize_zip(row[zc], row[cc]), axis=1)

    # Enforce most common street per Domain
    if "Domain" in mapping and "Street" in mapping:
        dom_col = mapping["Domain"]
        street_col = mapping["Street"]
        chosen = {}
        for domain, group in df.groupby(dom_col):
            non_empty = [s for s in group[street_col].astype(str) if s and s.strip()]
            chosen_val = Counter(non_empty).most_common(1)[0][0] if non_empty else ""
            chosen[domain] = chosen_val
        df[street_col] = df[dom_col].map(lambda d: chosen.get(d, ""))

    return df


if __name__ == "__main__":
    # Quick test
    sample = pd.DataFrame({
        'Company Name': ['ACME, Inc.', 'acme inc'],
        'First Name': ['john', 'MARY'],
        'Last Name': ['doe', "o'neil"],
        'Street': ['123 N Main St.', '123 north main st'],
        'City': ['new york', 'NEW YORK'],
        'Domain': ['acme.com', 'acme.com'],
        'Country': ['United States', 'US'],
        'Zip Code': ['2345', '123456']
    })
    print(apply_properization(sample))
