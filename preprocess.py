# preprocess.py
import re
import pandas as pd
from collections import Counter

# Direction tokens that should remain uppercase
DIRECTION_TOKENS = {"N", "S", "E", "W", "NE", "NS", "NW", "SN", "SE", "SW", "ES", "EW", "EN", "WE", "WS", "WN"}

# Common street types mapping to proper case
STREET_TYPE_MAP = {
    "st": "St", "rd": "Rd", "ave": "Ave", "blvd": "Blvd",
    "ln": "Ln", "dr": "Dr", "hwy": "Hwy", "pkwy": "Pkwy",
    "cir": "Cir", "ct": "Ct", "pl": "Pl",
}

PUNCT_TO_REMOVE = r"[,\.':()\;]"
ORDINAL_RE = re.compile(r"(\d+)(st|nd|rd|th)", flags=re.IGNORECASE)


def _canonical_col(df, target_name):
    """Find a column in df that matches target_name ignoring case/spacing/punctuation."""
    tn = re.sub(r"\s+", "", target_name).lower()
    for c in df.columns:
        cc = re.sub(r"\s+", "", c).lower()
        if cc == tn:
            return c
    return None


def _clean_basic_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(PUNCT_TO_REMOVE, "", s)  # remove punctuation
    s = s.replace('-', ' ')  # replace hyphens with spaces
    s = re.sub(r"\s+", " ", s)  # collapse multiple spaces
    return s.strip()


def _propercase_token(token: str) -> str:
    t = token.strip()
    if not t:
        return t
    if t.upper() in DIRECTION_TOKENS:
        return t.upper()
    if len(t) == 2 and t.isalpha():
        return t.upper()
    m = ORDINAL_RE.fullmatch(t)
    if m:
        num, suf = m.groups()
        return f"{num}{suf.lower()}"
    return t.title()


def _normalize_street_token(t: str) -> str:
    """Convert common street types to proper casing if found in STREET_TYPE_MAP."""
    lower = t.lower()
    if lower in STREET_TYPE_MAP:
        return STREET_TYPE_MAP[lower]
    return t


def properize_street(s: str) -> str:
    s = _clean_basic_text(s)
    if not s:
        return s
    tokens = s.split()
    tokens = [_propercase_token(t) for t in tokens]
    tokens = [_normalize_street_token(t) for t in tokens]
    return " ".join(tokens)


def _normalize_zip(zip_code: str, country: str) -> str:
    """Normalize ZIP code for US addresses (pad 4-digit with leading zero)."""
    if pd.isna(zip_code):
        return zip_code
    if not country:
        return zip_code

    if str(country).strip().lower() not in ["united states", "us"]:
        return zip_code  # only process US

    z = str(zip_code).strip()
    if not z.isdigit():
        return z  # leave as-is
    if len(z) == 4:
        return f"`0{z}`"  # pad and wrap in backticks
    if len(z) != 5:
        return z  # invalid length, leave as-is (can be highlighted later)
    return z


def apply_properization(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new DataFrame with requested columns normalized."""
    df = df.copy()

    # Map canonical column names
    mapping = {}
    for target in ["Company Name", "First Name", "Last Name", "Street", "City", "Domain", "Country", "Zip Code"]:
        found = _canonical_col(df, target)
        if found:
            mapping[target] = found

    # Company, First, Last, City -> clean + title-case
    for col in ["Company Name", "First Name", "Last Name", "City"]:
        if col in mapping:
            c = mapping[col]
            df[c] = df[c].fillna("").astype(str).apply(_clean_basic_text).apply(lambda s: s.title())

    # Street -> properize
    if "Street" in mapping:
        c = mapping["Street"]
        df[c] = df[c].fillna("").astype(str).apply(properize_street)

    # Zip code normalization
    if "Zip Code" in mapping and "Country" in mapping:
        zc = mapping["Zip Code"]
        cc = mapping["Country"]
        df[zc] = df.apply(lambda row: _normalize_zip(row[zc], row[cc]), axis=1)

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
