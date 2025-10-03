# utils/properization.py
import pandas as pd
import re
from collections import Counter
import openpyxl
from openpyxl.styles import PatternFill

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

# 🔑 Robust PO Box detection (supports P.O.Box, Pobox, Box 123, Post Office Box 123)
POBOX_RE = re.compile(
    r"\bP\.?\s*O\.?\s*Box\b|"       # P.O. Box
    r"\bPobox\b|"                   # Pobox
    r"\bPost\s*Office\s*Box\b|"     # Post Office Box
    r"\bBox\s*\d+\b",               # Box 123
    flags=re.IGNORECASE
)


def _canonical_col(df, target_name: str):
    tn = re.sub(r"\s+", "", target_name).lower()
    for c in df.columns:
        cc = re.sub(r"\s+", "", c).lower()
        if cc == tn:
            return c
    return None


def _clean_basic_text(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).strip()
    value = re.sub(PUNCT_TO_REMOVE, "", value)
    value = value.replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_text(value: str) -> str:
    value = _clean_basic_text(value)
    return value.title() if value else ""


def _propercase_street_token(token: str) -> str:
    if not token:
        return token
    upper = token.upper()

    if upper in DIRECTION_TOKENS:
        return upper

    if upper.lower() in STREET_TYPE_MAP:
        return STREET_TYPE_MAP[upper.lower()]

    m = ORDINAL_RE.fullmatch(token)
    if m:
        num, suf = m.groups()
        return f"{num}{suf.lower()}"

    return token.title()


def clean_street(value: str) -> str:
    value = _clean_basic_text(value)
    if not value:
        return value
    tokens = [_propercase_street_token(tok) for tok in value.split()]
    return " ".join(tokens)


def _properize_zip(zipcode: str, country: str) -> str:
    if pd.isna(zipcode):
        return ""
    zipcode = str(zipcode).strip()
    country = str(country).strip().upper()

    if country in ("US", "UNITED STATES", "USA", "UNITED STATES OF AMERICA", "United States", "United States Of America"):
        if ZIPCODE_RE.match(zipcode):
            if len(zipcode) == 4:
                return f"`0{zipcode}`"
            elif len(zipcode) == 5:
                return zipcode
            else:
                return f"INVALID:{zipcode}"
        else:
            return f"INVALID:{zipcode}"
    return zipcode


def _detect_pobox(street: str) -> bool:
    """Return True if PO Box is detected in the street"""
    if not street or pd.isna(street):
        return False
    # 🔑 Use original street, no cleaning
    return bool(POBOX_RE.search(street))


def apply_properization(df: pd.DataFrame, enforce_common_street: bool = True) -> pd.DataFrame:
    """Apply properization rules to DataFrame.
    enforce_common_street: if True, will enforce most common street, city, state, zip, country per Domain
    """
    df = df.copy()

    mapping = {}
    for target in ["Company Name", "First Name", "Last Name", "Street", "City", "State", "Domain", "Country", "Zip Code"]:
        found = _canonical_col(df, target)
        if found:
            mapping[target] = found

    for col in ["Company Name", "First Name", "Last Name", "City", "State", "Country"]:
        if col in mapping:
            c = mapping[col]
            df[c] = df[c].fillna("").astype(str).apply(clean_text)

    if "Street" in mapping:
        c = mapping["Street"]
        # --- Detect PO Box using original street value before cleaning ---
        df["Street_POBox_Flag"] = df[c].fillna("").astype(str).apply(_detect_pobox)
        # Then clean street for display
        df[c] = df[c].fillna("").astype(str).apply(clean_street)

    if "Zip Code" in mapping and "Country" in mapping:
        zc = mapping["Zip Code"]
        cc = mapping["Country"]
        df[zc] = df.apply(lambda row: _properize_zip(row[zc], row[cc]), axis=1)

    # 🔑 Apply common values per domain
    if enforce_common_street and "Domain" in mapping:
        dom_col = mapping["Domain"]
        columns_to_enforce = ["Street", "City", "State", "Zip Code", "Country"]
        for col_name in columns_to_enforce:
            if col_name in mapping:
                col = mapping[col_name]
                chosen = {}
                for domain, group in df.groupby(dom_col):
                    non_empty = [s for s in group[col].astype(str) if s and s.strip()]
                    chosen_val = Counter(non_empty).most_common(1)[0][0] if non_empty else ""
                    chosen[domain] = chosen_val
                df[col] = df[dom_col].map(lambda d: chosen.get(d, ""))

    return df


def apply_pobox_coloring(excel_path: str, df: pd.DataFrame):
    """
    Color the 'Street' column if it contains PO Box.
    Yellow color fill.
    """
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active

    fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")  # Yellow

    col_idx = None
    for idx, cell in enumerate(ws[1], 1):
        if cell.value == "Street":
            col_idx = idx
            break
    if col_idx is None:
        return

    for row_num, flag in enumerate(df.get("Street_POBox_Flag", []), 2):
        if flag:
            ws.cell(row=row_num, column=col_idx).fill = fill

    wb.save(excel_path)


if __name__ == "__main__":
    sample = pd.DataFrame({
        'Company Name': ['ACME, Inc.', 'acme inc'],
        'First Name': ['john', 'MARY'],
        'Last Name': ['doe', "o'neil"],
        'Street': [
            '123 N Main St.',
            'Pobox 79 Taman Perindustrian Lot 60334 Persiaran 3 Bukit Rahman Putra'
        ],
        'City': ['new york', 'NEW YORK'],
        'State': ['NY', 'ny'],
        'Domain': ['acme.com', 'acme.com'],
        'Country': ['United States', 'US'],
        'Zip Code': ['2345', '123456']
    })
    result = apply_properization(sample, enforce_common_street=True)
    print(result)

    # Example usage of Excel coloring
    # result.to_excel("sample.xlsx", index=False)
    # apply_pobox_coloring("sample.xlsx", result)
