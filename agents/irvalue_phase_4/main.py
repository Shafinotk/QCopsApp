# agents/irvalue_phase_4/main.py
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import chardet
from tqdm import tqdm

from search_utils import search_zoominfo, search_linkedin
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

def revenue_queries(company_name: str, domain: str, country: str):
    return [
        f'{domain} {company_name} {country} revenue zoominfo',
        f'{domain} {company_name} {country} company revenue zoominfo',
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

# ---------- Normalization helpers ----------
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
    return cn in t if cn else False

def _domain_mentioned(domain: str, r: dict) -> bool:
    joined = ((r.get("href") or "") + " " + (r.get("title") or "") + " " + (r.get("body") or "")).lower()
    return domain.lower() in joined if domain else False

# ---------- Scoring ----------
def _score_result(r, company_name, domain, extractor):
    score = 0
    title, body, href = r.get("title") or "", r.get("body") or "", r.get("href") or ""
    text = title + " " + body
    if _company_matches(company_name, text):
        score += 2
    if _domain_mentioned(domain, r):
        score += 3
    if extractor(body) or extractor(title):
        score += 1
    return score

# ---------- Extraction pipelines ----------
def _best_result(results, company_name, domain, extractor, parser,
                 enforce_company_match=False, validator=None, emp_val=None):
    candidates = []
    best = None
    best_score = -1
    best_similarity = -1

    for r in results:
        title, body, href = r.get("title") or "", r.get("body") or "", r.get("href") or ""
        text_block = f"{title} {body}"

        if enforce_company_match and not (_company_matches(company_name, text_block) or _company_matches(company_name, href)):
            continue

        score = _score_result(r, company_name, domain, extractor)
        val = extractor(body) or extractor(title)
        if not val:
            continue
        parsed_val = parser(val)
        if not parsed_val:
            continue

        similarity = len(company_name) if company_name.lower() in text_block.lower() else 0
        candidates.append((val, parsed_val, href, score, similarity))

        if score > best_score or (score == best_score and similarity > best_similarity):
            best = (val, parsed_val, href)
            best_score = score
            best_similarity = similarity

    if not candidates:
        return None, None, False

    # Apply RPE validator if available
    if validator and emp_val:
        emp_num = parse_employees(emp_val)
        valid = []
        for (orig, parsed, h, s, sim) in candidates:
            parsed_num = parse_revenue(parsed)
            if emp_num and parsed_num and validator(parsed_num, emp_num):
                valid.append((orig, parsed_num, h))
        if valid:
            return valid[0][0], valid[0][2], False
        else:
            return best[0], best[2], True

    return best[0], best[2], False

def extract_field_via_queries(queries, company_name, domain, extractor, parser, search_func, enforce_company_match=False, validator=None, emp_val=None):
    aggregated = []
    seen_hrefs = set()
    for q in queries:
        results = search_func(q, max_results=50)
        if not results:
            continue
        for r in results:
            href = r.get("href") or ""
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            aggregated.append(r)
    if not aggregated:
        return None, None, False
    return _best_result(aggregated, company_name, domain, extractor, parser, enforce_company_match=enforce_company_match, validator=validator, emp_val=emp_val)

# ---------- Worker ----------
def find_company_info(domain, country, company):
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
        emp_val=emp_val
    )
    ind_val, _, _ = extract_field_via_queries(
        industry_queries(company, domain, country),
        company, domain,
        extract_industry, parse_industry_value,
        search_linkedin,
        enforce_company_match=True
    )

    # --- Ensure numeric parsing before saving ---
    emp_num = parse_employees(emp_val)
    rev_num = parse_revenue(rev_val)

    return emp_val, rev_val, ind_val, flagged, emp_num, rev_num

# ---------- Async orchestration with progress ----------
async def process_unique_domain(loop, executor, domain, country, company):
    return await loop.run_in_executor(executor, find_company_info, domain, country, company)

async def irvalue_logic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Domain", "Country", "Company Name"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ""

    # Precompute RPE range (fixed now at 60k–9M)
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
