# agents/irvalue_phase_4/main.py
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import chardet
from tqdm import tqdm
import numpy as np
from urllib.parse import urlparse, unquote

from search_utils import search_zoominfo, search_linkedin, fetch_html
from extract_utils import (
    extract_employees, extract_revenue, extract_industry,
    format_employee_value, format_revenue_value, parse_industry_value,
    parse_employees, parse_revenue, is_valid_rpe, set_rpe_range_from_data
)

# ---------- Query builders ----------
def employee_queries(company_name: str, domain: str, country: str):
    return [
        f'{company_name} {domain} {country} employee size site:zoominfo.com',
        f'{company_name} {country} employees site:zoominfo.com',
        f'{domain} {country} employees site:zoominfo.com',
        f'{company_name} employees site:zoominfo.com',
    ]

def revenue_queries(company_name, domain, country):
    return [
        f'{domain} {company_name} {country} revenue site:zoominfo.com',
        f'{domain} {company_name} {country} company revenue site:zoominfo.com',
        f'{company_name} {domain} {country} revenue site:zoominfo.com',
        f'{company_name} {country} annual revenue site:zoominfo.com',
        f'{domain} {country} revenue site:zoominfo.com',
        f'{company_name} revenue site:zoominfo.com',
    ]

def industry_queries(company_name: str, domain: str, country: str):
    name_q = f'"{company_name}"' if company_name else company_name
    return [
        f'{name_q} site:linkedin.com/company/about',
        f'{name_q} site:linkedin.com/company "Industry"',
        f'{name_q} {domain} site:linkedin.com/company',
        f'{name_q} {country} site:linkedin.com/company',
    ]

# ---------- Helpers ----------
LEGAL_SUFFIXES = r'(?:inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|gmbh|s\.p\.a|s\.a|bv)'

def _normalize_company(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = s.replace("&", " and ")
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(rf'\b{LEGAL_SUFFIXES}\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _company_matches(company_name: str, text: str) -> bool:
    cn = _normalize_company(company_name)
    t = _normalize_company(text or "")
    # require whole token match (substring only if safe)
    if not cn:
        return False
    # check tokens presence
    cn_tokens = set(cn.split())
    t_tokens = set(t.split())
    # require at least half of company tokens to appear (prevents accidental short substring matches)
    if not cn_tokens:
        return False
    common = cn_tokens.intersection(t_tokens)
    return (len(common) >= max(1, len(cn_tokens)//2))

def _domain_mentioned(domain: str, r: dict) -> bool:
    joined = ((r.get("href") or "") + " " + (r.get("title") or "") + " " + (r.get("body") or "")).lower()
    return domain.lower() in joined if domain else False

def is_zoominfo_profile(href: str) -> bool:
    if not href:
        return False
    h = href.lower()
    return ("zoominfo.com" in h) and (
        "/company/" in h or "/c/" in h or "/profile/" in h or "/company-profile" in h
    )

def url_slug_matches_company(company_name: str, href: str) -> bool:
    """Look at URL path slug and check if company tokens appear there."""
    if not href or not company_name:
        return False
    try:
        p = urlparse(href)
        path = unquote(p.path)
        slug = path.strip("/").replace("-", " ").replace("_", " ").lower()
        return _company_matches(company_name, slug)
    except Exception:
        return False

def company_near_revenue(text: str, company_name: str, revenue_substring: str, window_chars:int=80) -> bool:
    """Return True if company_name appears within window_chars around revenue_substring in text."""
    if not text or not company_name or not revenue_substring:
        return False
    idx = text.lower().find(revenue_substring.lower())
    if idx == -1:
        return False
    start = max(0, idx - window_chars)
    end = idx + len(revenue_substring) + window_chars
    snippet = text[start:end]
    return _company_matches(company_name, snippet)

# ---------- Extraction pipelines ----------
def _best_result(results, company_name, domain, extractor, parser,
                 enforce_company_match=False, validator=None, emp_val=None, debug=False):
    candidates = []

    for r in results:
        title, body, href = r.get("title") or "", r.get("body") or "", r.get("href") or ""
        text_block = f"{title} {body}"

        if enforce_company_match and not (_company_matches(company_name, text_block) or _company_matches(company_name, href)):
            continue

        if extractor == extract_revenue:
            # Only ZoomInfo pages (per requirement)
            if "zoominfo.com" not in (href or "").lower():
                continue

            # Try snippet first (DDG title/body)
            revenue_val_snippet = extractor(body) or extractor(title)

            # If available, fetch the zoominfo page HTML (page parsing is much more accurate)
            revenue_val_html = None
            try:
                html = fetch_html(href, timeout=8)
                if html:
                    # run extractor on page HTML (this will try to find revenue label on the page)
                    revenue_val_html = extractor(html)
            except Exception:
                revenue_val_html = None

            # Prefer revenue found on the page HTML (more authoritative) otherwise use snippet
            revenue_val = revenue_val_html or revenue_val_snippet
            if not revenue_val:
                continue

            # parsed numeric value
            parsed_val = None
            try:
                parsed_val = parse_revenue(revenue_val)
            except Exception:
                parsed_val = None
            if not parsed_val:
                # If parse_revenue returned None (e.g. odd formats), still keep raw candidate but mark approx
                # We'll still include it so fallback can pick it.
                parsed_val = None

            # Matching checks: company token presence and domain mention and URL slug
            company_match = _company_matches(company_name, text_block) or url_slug_matches_company(company_name, href)
            domain_match = _domain_mentioned(domain, r)
            profile_flag = is_zoominfo_profile(href)

            # proximity check: company name near revenue substring (if snippet provided)
            prox_ok = False
            if revenue_val:
                prox_ok = company_near_revenue(text_block, company_name, revenue_val, window_chars=80)
                # also check HTML snippet if present
                if not prox_ok and revenue_val_html and html:
                    prox_ok = company_near_revenue(html, company_name, revenue_val_html, window_chars=200)

            # require either company_match or domain_match or slug match or profile_flag (tighten)
            if not (company_match or domain_match or profile_flag):
                # skip likely wrong matches (too loose)
                continue

            # Score: profile and domain and company_match and proximity
            score = 0
            if profile_flag:
                score += 4
            if domain_match:
                score += 3
            if company_match:
                score += 3
            if prox_ok:
                score += 2

            # approx flag if revenue contains '<' or '>'
            approx = isinstance(revenue_val, str) and revenue_val.strip().startswith(("<", ">"))

            candidates.append({
                "val_raw": revenue_val,
                "parsed": parsed_val,
                "href": href,
                "score": score,
                "approx": approx
            })

        else:
            # non-revenue extraction (employees / industry)
            val = extractor(body) or extractor(title)
            if not val:
                continue
            parsed_val = parser(val)
            if not parsed_val:
                continue
            candidates.append({"val_raw": val, "parsed": parsed_val, "href": href, "score": 1, "approx": False})

    if not candidates:
        return None, None, False

    # For revenue: sort by score, prefer non-approx before approx, and then by parsed value (desc)
    if extractor == extract_revenue:
        candidates.sort(key=lambda c: (c["score"], not c["approx"], c["parsed"] or 0), reverse=True)

        # RPE validation: if employee value present, prefer numeric candidates that pass RPE
        if validator and emp_val:
            emp_num = parse_employees(emp_val)
            if isinstance(emp_num, int):
                # first try non-approx numeric candidates that pass RPE
                for c in candidates:
                    if (not c["approx"]) and (c["parsed"] is not None) and validator(c["parsed"], emp_num):
                        return c["val_raw"], c["href"], False
                # if none pass, consider approx numeric candidates if they might still pass as bounds
                for c in candidates:
                    if c["approx"] and (c["parsed"] is not None):
                        # treat '<X' as upper bound. We'll accept approx only if using the numeric value would not produce impossible RPE
                        # Conservative: accept approx only if validator(parsed_value, emp_num) is True (i.e., even upper bound passes)
                        if validator(c["parsed"], emp_num):
                            return c["val_raw"], c["href"], False
                # no candidate passed RPE strictly -> return best candidate but flagged True
                top = candidates[0]
                return top["val_raw"], top["href"], True

    # Default return: take best candidate
    # sort by score descending for non-revenue or fallback
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[0]
    return top["val_raw"], top["href"], False


def extract_field_via_queries(queries, company_name, domain, extractor, parser, search_func,
                              enforce_company_match=False, validator=None, emp_val=None, debug=False):
    aggregated = []
    seen_hrefs = set()
    for q in queries:
        results = search_func(q, max_results=100)   # increased results
        if not results:
            continue
        for r in results:
            href = r.get("href") or ""
            # minor normalization to avoid fragments causing duplicates
            href_norm = href.split("#")[0]
            if href_norm in seen_hrefs:
                continue
            seen_hrefs.add(href_norm)
            aggregated.append(r)
    if not aggregated:
        return None, None, False
    return _best_result(aggregated, company_name, domain, extractor, parser,
                        enforce_company_match=enforce_company_match, validator=validator, emp_val=emp_val, debug=debug)

# ---------- Worker ----------
def find_company_info(domain, country, company, debug=False):
    emp_val, _, _ = extract_field_via_queries(
        employee_queries(company, domain, country),
        company, domain,
        extract_employees, format_employee_value,
        search_zoominfo
    )

    rev_val, _, flagged = extract_field_via_queries(
        revenue_queries(company, domain, country),
        company, domain,
        extract_revenue, format_revenue_value,
        search_zoominfo,
        validator=is_valid_rpe,
        emp_val=emp_val,
        debug=debug
    )

    ind_val, _, _ = extract_field_via_queries(
        industry_queries(company, domain, country),
        company, domain,
        extract_industry, parse_industry_value,
        search_linkedin,
        enforce_company_match=True
    )

    emp_num = parse_employees(emp_val)
    rev_num = parse_revenue(rev_val) if rev_val else None

    return emp_val, rev_val, ind_val, flagged, emp_num, rev_num

# ---------- Async orchestration ----------
async def process_unique_domain(loop, executor, domain, country, company):
    return await loop.run_in_executor(executor, find_company_info, domain, country, company)

async def irvalue_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Domain", "Country", "Company Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    if "Revenue_per_Employee" in df.columns:
        set_rpe_range_from_data(df["Revenue_per_Employee"])

    domain_jobs = {row["Domain"].lower(): (row["Domain"], row["Country"], row["Company Name"])
                   for _, row in df.iterrows() if row.get("Domain")}

    loop = asyncio.get_running_loop()
    results_by_domain = {}

    with ThreadPoolExecutor(max_workers=12) as executor:
        tasks = [(key, asyncio.create_task(process_unique_domain(loop, executor, dom, country, company)))
                 for key, (dom, country, company) in domain_jobs.items()]

        for key, task in tqdm(tasks, total=len(tasks), desc="IRValue Progress"):
            emp, rev, ind, flagged, emp_num, rev_num = await task
            results_by_domain[key] = (emp, rev, ind, flagged, emp_num, rev_num)

    df["discovered_employees_raw"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[0])
    df["discovered_revenue_raw"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[1])
    df["discovered_industry"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[2])
    df["flagged_rpe"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[3])

    # --- Numeric columns ---
    df["discovered_employees"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[4])
    df["discovered_revenue"] = df["Domain"].apply(lambda d: results_by_domain.get((d or "").lower(), (None, None, None, False, None, None))[5])

    return df

# ---------- CLI Entry ----------
def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run IRValue Agent")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", required=True, help="Path to save output CSV")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw = f.read(100000)
        encoding = chardet.detect(raw)["encoding"] or "utf-8"

    df = pd.read_csv(args.input, encoding=encoding, dtype=str, keep_default_na=False)
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    result_df = asyncio.run(irvalue_logic(df))
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ IRValue processing complete. Output saved to {args.output}")

if __name__ == "__main__":
    run_cli()
