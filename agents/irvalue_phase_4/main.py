import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import chardet
import logging
from tqdm import tqdm
from typing import Optional

# =========================================================
# 🔗 LOCAL IMPORTS (cleaned to avoid circular imports)
# =========================================================
from .search_utils import search_zoominfo, search_linkedin, fetch_linkedin_industry
from .extract_utils import (
    extract_employees,
    extract_revenue,
    extract_industry,
    format_employee_value,
    format_revenue_value,
    parse_industry_value,
    is_valid_rpe,
    employees_to_number,
    revenue_to_number,
    snippet_similarity_score,
)
from .file_first_utils import get_values_from_file, load_reference_file

# =========================================================
# 🧠 LOGGING
# =========================================================
logger = logging.getLogger("irvalue.main")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# =========================================================
# 🔍 QUERY BUILDERS
# =========================================================
def employee_queries(company_name: str, domain: str, country: str):
    return [
        f'{company_name} {domain} {country} employee size site:zoominfo.com',
        f'{company_name} {country} employees site:zoominfo.com',
        f'{domain} {country} employees site:zoominfo.com',
        f'{company_name} employees site:zoominfo.com',
    ]

def revenue_queries(company_name: str, domain: str, country: str):
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

# =========================================================
# 🧹 NORMALIZATION HELPERS
# =========================================================
LEGAL_SUFFIXES = r'(?:inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|gmbh|s\.p\.a|s\.a|bv)'

def _normalize_company(s: str) -> str:
    if not s:
        return ""
    s = s.lower().replace("&", " and ")
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(rf'\b{LEGAL_SUFFIXES}\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _company_matches(company_name: str, text: str) -> bool:
    cn = _normalize_company(company_name)
    if not cn:
        return False
    t = _normalize_company(text)
    return cn in t

def _domain_mentioned(domain: str, r: dict) -> bool:
    if not domain:
        return False
    dom = domain.lower()
    joined = ((r.get("href") or "") + " " + (r.get("title") or "") + " " + (r.get("body") or "")).lower()
    return dom in joined

# =========================================================
# 📊 SNIPPET SCORING
# =========================================================
def _score_result(r, company_name, domain, extractor):
    body = r.get("body") or ""
    title = r.get("title") or ""
    href = r.get("href") or ""

    candidates = [title] + re.split(r'[\n\.]', body)
    best_snippet = ""
    best_score = -1

    field_keywords = []
    if extractor.__name__ == "extract_revenue":
        field_keywords = ["revenue", "sales", "turnover"]
    elif extractor.__name__ == "extract_employees":
        field_keywords = ["employee", "staff", "workforce", "size"]
    elif extractor.__name__ == "extract_industry":
        field_keywords = ["industry", "sector", "field"]

    for snippet in candidates:
        snippet = snippet.strip()
        if not snippet:
            continue

        if not (_company_matches(company_name, snippet) or (domain and domain.lower() in snippet.lower())):
            continue

        score = snippet_similarity_score(company_name, snippet)

        if "zoominfo" in href.lower():
            score += 1.0
        if any(k in snippet.lower() for k in field_keywords):
            score += 0.7
        if "revenue" in snippet.lower() and "employees" in snippet.lower():
            score += 0.3
        if len(re.findall(r"\d+", snippet)) > 3:
            score -= 0.5
        if extractor(snippet):
            score += 1.2

        if score > best_score:
            best_score = score
            best_snippet = snippet

    return best_snippet, best_score

# =========================================================
# 🏆 SELECT BEST RESULT
# =========================================================
def _best_result(results, company_name, domain, extractor, parser,
                 enforce_company_match=True, prefer_html=False):
    best_val, best_href, best_snippet = None, None, None
    best_score = -1

    for r in results:
        href = r.get("href") or ""
        if extractor.__name__ in ["extract_employees", "extract_revenue"]:
            if "zoominfo.com" not in href.lower():
                continue
        elif extractor.__name__ == "extract_industry":
            if "linkedin.com/company" not in href.lower():
                continue

        snippet, score = _score_result(r, company_name, domain, extractor)
        if not snippet:
            continue

        if enforce_company_match and not _company_matches(company_name, snippet):
            continue

        val = None
        if prefer_html and "linkedin.com/company" in href:
            try:
                val = fetch_linkedin_industry(href)
            except Exception:
                pass
        if not val:
            val = extractor(snippet)
        if not val:
            continue

        parsed = parser(val)
        if not parsed:
            continue

        if score > best_score:
            best_val, best_href, best_score, best_snippet = parsed, href, score, snippet

    return best_val, best_href, best_snippet

# =========================================================
# 🔎 FIELD EXTRACTION VIA MULTIPLE QUERIES
# =========================================================
def extract_field_via_queries(queries, company_name, domain, extractor, parser,
                              search_func, enforce_company_match=True,
                              prefer_html=False, per_query_max=30):
    aggregated, seen = [], set()
    for q in queries:
        results = search_func(q, max_results=per_query_max)
        if not results:
            continue
        for r in results:
            href = r.get("href")
            if not href or href in seen:
                continue
            seen.add(href)
            aggregated.append(r)
    if not aggregated:
        return None, None, None
    return _best_result(aggregated, company_name, domain, extractor, parser,
                        enforce_company_match, prefer_html)

# =========================================================
# ⚙️ COMPANY INFO AGGREGATOR
# =========================================================
REFERENCE_FILE_PATH = r"C:\desktop\thinkogent Auto\apps\qcopsappNEW2\qcopsappnew\agents\irvalue_phase_4\IR Values master file.xlsx"
reference_df = load_reference_file(REFERENCE_FILE_PATH)

def find_company_info(domain, country, company):
    emp_val, rev_val, ind_val = get_values_from_file(domain, reference_df)

    if not emp_val:
        emp_val, _, _ = extract_field_via_queries(
            employee_queries(company, domain, country),
            company, domain,
            extract_employees, format_employee_value,
            search_zoominfo
        )

    if not rev_val:
        rev_val, _, _ = extract_field_via_queries(
            revenue_queries(company, domain, country),
            company, domain,
            extract_revenue, format_revenue_value,
            search_zoominfo
        )

    if not ind_val:
        ind_val, _, _ = extract_field_via_queries(
            industry_queries(company, domain, country),
            company, domain,
            extract_industry, parse_industry_value,
            search_linkedin,
            enforce_company_match=True,
            prefer_html=True
        )

    logger.info(
        f"\nCompany: {company}, Domain: {domain}, Country: {country}\n"
        f"Employees: {emp_val}\nRevenue: {rev_val}\nIndustry: {ind_val}"
    )

    flagged = False
    try:
        emp_num = employees_to_number(emp_val)
        rev_num = revenue_to_number(rev_val)
        if emp_num and rev_num:
            flagged = not is_valid_rpe(rev_num, emp_num)
    except Exception:
        flagged = False

    return emp_val, rev_val, ind_val, flagged

# =========================================================
# ⚡️ ASYNC WRAPPERS
# =========================================================
async def process_unique_domain(loop, executor, domain, country, company):
    return await loop.run_in_executor(executor, find_company_info, domain, country, company)

async def irvalue_logic(df: pd.DataFrame, debug=False):
    df = df.copy()
    for col in ["Domain", "Country", "Company Name"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str).str.strip()

    jobs = {
        (d.lower(), n.lower(), c.lower()): (d, c, n)
        for d, c, n in zip(df["Domain"], df["Country"], df["Company Name"])
        if str(d).strip()
    }

    loop = asyncio.get_running_loop()
    results = {}

    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {}
        for key, (dom, country, company) in jobs.items():
            emp_val, rev_val, ind_val = get_values_from_file(dom, reference_df)

            if not emp_val or not rev_val or not ind_val:
                from functools import partial
                task = loop.run_in_executor(executor, partial(find_company_info, dom, country, company))
                tasks[key] = task
            else:
                fut = asyncio.Future()
                flagged = False
                try:
                    emp_num = employees_to_number(emp_val)
                    rev_num = revenue_to_number(rev_val)
                    if emp_num and rev_num:
                        flagged = not is_valid_rpe(rev_num, emp_num)
                except Exception:
                    flagged = False
                fut.set_result((emp_val, rev_val, ind_val, flagged))
                tasks[key] = fut

        for key, task in tqdm(tasks.items(), total=len(tasks), desc="IRValue Progress"):
            try:
                emp, rev, ind, flagged = await task
            except Exception as e:
                logger.exception("Task failed for %s: %s", key, e)
                emp, rev, ind, flagged = None, None, None, False
            results[key] = (emp, rev, ind, flagged)

    def _get(domain, company, country):
        return results.get((domain.lower(), company.lower(), country.lower()), (None, None, None, False))

    df["discovered_employees"] = df.apply(lambda r: _get(r["Domain"], r["Company Name"], r["Country"])[0], axis=1)
    df["discovered_revenue"] = df.apply(lambda r: _get(r["Domain"], r["Company Name"], r["Country"])[1], axis=1)
    df["discovered_industry"] = df.apply(lambda r: _get(r["Domain"], r["Company Name"], r["Country"])[2], axis=1)
    df["flagged_rpe"] = df.apply(lambda r: _get(r["Domain"], r["Company Name"], r["Country"])[3], axis=1)

    return df

# =========================================================
# 🧾 CLI ENTRYPOINT
# =========================================================
def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Run IRValue Agent (sub or standalone)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        raw = f.read(100000)
        encoding = chardet.detect(raw)["encoding"] or "utf-8"

    df = pd.read_csv(args.input, encoding=encoding, dtype=str, keep_default_na=False)
    result_df = asyncio.run(irvalue_logic(df, debug=args.debug))
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"✅ IRValue results saved to {args.output}")


if __name__ == "__main__":
    run_cli()
