# agents/linkedin_agent/file_handler.py
import pandas as pd
import io
import chardet
from linkedin_finder import find_link
from typing import Optional
import re

def _normalize_colname(s: str) -> str:
    return re.sub(r"\s+|[_\-\.]", "", str(s).lower())

def _find_column(df: pd.DataFrame, target: str) -> Optional[str]:
    """Find a column in df that matches target ignoring case/spacing/punctuation."""
    tnorm = _normalize_colname(target)
    for c in df.columns:
        if _normalize_colname(c) == tnorm:
            return c
    return None

def process_csv(file_bytes: bytes) -> bytes:
    """
    Process uploaded CSV bytes, add 'linkedin_link_found' column (if missing),
    and return CSV bytes.
    """
    # Detect encoding
    enc_result = chardet.detect(file_bytes)
    encoding = enc_result.get("encoding") or "utf-8"
    print(f"[file_handler] Detected encoding: {encoding}")

    # Read CSV
    df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]

    # Find canonical column names
    first_col = _find_column(df, "First Name") or "First Name"
    last_col = _find_column(df, "Last Name") or "Last Name"
    company_col = _find_column(df, "Company Name") or "Company Name"
    title_col = _find_column(df, "Title") or "Title"
    domain_col = _find_column(df, "Domain") or "Domain"
    email_col = _find_column(df, "Email") or "Email"

    # Ensure columns exist
    for col in [first_col, last_col, company_col, title_col, domain_col, email_col]:
        if col not in df.columns:
            df[col] = ""

    # Add output column if missing
    out_col = "linkedin_link_found"
    if out_col not in df.columns:
        df[out_col] = ""

    total = len(df)
    found_count = 0
    print(f"[file_handler] Processing {total} rows (cols: {first_col}, {last_col}, {company_col}, {title_col}, {domain_col}, {email_col})")

    for idx, row in df.iterrows():
        try:
            first = (row.get(first_col) or "").strip()
            last = (row.get(last_col) or "").strip()
            company = (row.get(company_col) or "").strip()
            title = (row.get(title_col) or "").strip()
            domain = (row.get(domain_col) or "").strip()
            email = (row.get(email_col) or "").strip()

            link = find_link(first, last, company, title, domain, email)
            if link:
                df.at[idx, out_col] = link
                found_count += 1

            if idx and idx % 50 == 0:
                print(f"[file_handler] progress: {idx}/{total} rows, found {found_count} links so far.")
        except Exception as e:
            print(f"[file_handler] Error processing row {idx}: {e}")
            continue

    print(f"[file_handler] Done. Found {found_count} links out of {total} rows.")

    # Write CSV back to bytes
    output_buffer = io.StringIO()
    df.to_csv(output_buffer, index=False, encoding="utf-8")
    return output_buffer.getvalue().encode("utf-8")
