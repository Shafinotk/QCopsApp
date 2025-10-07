# agents/irvalue_phase_4/main.py
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import chardet
from tqdm import tqdm
from urllib.parse import urlparse, unquote
from typing import Optional, Tuple, Any
import logging
import os, json, hashlib
from transformers import pipeline
from pathlib import Path


from .search_utils import search_zoominfo, fetch_industry_from_linkedin_about, fetch_html
from .extract_utils import (
    extract_employees, extract_revenue, extract_industry,
    format_employee_value, format_revenue_value, parse_industry_value,
    parse_employees, parse_revenue, is_valid_rpe, set_rpe_range_from_data, RANGE_DASH
)
# NEW: validation helpers
from .validation_utils import validate_domain, is_same_company, sanity_check, score_candidate


# Directory-based cache (persistent across runs)
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def _cache_path(domain: str, field: str) -> Path:
    key = hashlib.md5(f"{domain}_{field}".encode()).hexdigest()
    return CACHE_DIR / f"{key}.json"

def load_cache(domain: str, field: str):
    path = _cache_path(domain, field)
    if path.exists():
        try:
            return json.load(open(path, "r"))
        except Exception:
            return None
    return None

def save_cache(domain: str, field: str, data):
    try:
        json.dump(data, open(_cache_path(domain, field), "w"))
    except Exception:
        pass

# ---------- Lightweight LLM Extractor ----------
# flan-t5-small is ~300MB, much faster than base/large
try:
    llm_extractor = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        truncation=True,
        device=-1  # CPU-safe
    )
except Exception as e:
    llm_extractor = None
    print("⚠️ Could not load LLM extractor:", e)

def semantic_extract(text: str, field: str) -> Optional[str]:
    """
    Use LLM to extract a missing field.
    Field ∈ {'employees', 'revenue', 'industry'}
    """
    if not text or not llm_extractor:
        return None

    prompt = f"""
    Extract the {field} of the company from this text.
    If not found, output 'None'.
    Text: {text[:1200]}
    """

    try:
        response = llm_extractor(prompt, max_length=100, do_sample=False)
        val = response[0]['generated_text']
        if "none" in val.lower():
            return None
        return val.strip()
    except Exception:
        return None



# ---------- Logging ----------
logger = logging.getLogger("irvalue.main")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- Query Builders ----------
def employee_queries(company_name: str, domain: str, country: str) -> list[str]:
    return [
        f'{company_name} {domain} {country} employee size site:zoominfo.com',
        f'{company_name} {country} employees site:zoominfo.com',
        f'{domain} {country} employees site:zoominfo.com',
        f'{company_name} employees site:zoominfo.com',
    ]

def revenue_queries(company_name: str, domain: str, country: str) -> list[str]:
    return [
        f'{domain} {company_name} {country} revenue site:zoominfo.com',
        f'{domain} {company_name} {country} company revenue site:zoominfo.com',
        f'{company_name} {domain} {country} revenue site:zoominfo.com',
        f'{company_name} {country} annual revenue site:zoominfo.com',
        f'{domain} {country} revenue site:zoominfo.com',
        f'{company_name} revenue site:zoominfo.com',
    ]

def industry_queries(company_name: str, domain: str, country: str) -> list[str]:
    name_q = f'"{company_name}"' if company_name else company_name
    return [
        f'{name_q} site:linkedin.com/company/about',
        f'{name_q} site:linkedin.com/company "Industry"',
        f'{name_q} {domain} site:linkedin.com/company',
        f'{name_q} {country} site:linkedin.com/company',
    ]

# ---------- Helpers ----------
LEGAL_SUFFIXES = r'(?:inc|incorporated|llc|l\.l\.c|ltd|limited|corp|corporation|co|company|plc|gmbh|s\.p\.a|s\.a|bv)'

def _normalize_company(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower().replace("&", " and ")
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    s = re.sub(rf'\b{LEGAL_SUFFIXES}\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _company_matches(company_name: str, text: Optional[str]) -> bool:
    cn_tokens = set(_normalize_company(company_name).split())
    t_tokens = set(_normalize_company(text).split()) if text else set()
    if not cn_tokens:
        return False
    common = cn_tokens.intersection(t_tokens)
    return len(common) >= max(1, len(cn_tokens) // 2)

def _domain_mentioned(domain: str, r: dict) -> bool:
    combined_text = f"{r.get('href','')} {r.get('title','')} {r.get('body','')}".lower()
    return domain.lower() in combined_text if domain else False

def is_zoominfo_profile(href: str) -> bool:
    if not href:
        return False
    href_lower = href.lower()
    return "zoominfo.com" in href_lower and any(x in href_lower for x in ["/company/", "/c/", "/profile/", "/company-profile"])

def url_slug_matches_company(company_name: str, href: str) -> bool:
    if not href or not company_name:
        return False
    try:
        path = unquote(urlparse(href).path)
        slug = path.strip("/").replace("-", " ").replace("_", " ").lower()
        return _company_matches(company_name, slug)
    except Exception:
        return False

def company_near_revenue(text: str, company_name: str, revenue_substring: str, window_chars: int = 80) -> bool:
    if not text or not company_name or not revenue_substring:
        return False
    idx = text.lower().find(revenue_substring.lower())
    if idx == -1:
        return False
    start = max(0, idx - window_chars)
    end = idx + len(revenue_substring) + window_chars
    snippet = text[start:end]
    return _company_matches(company_name, snippet)

# ---------- Extraction Pipeline ----------
# (unchanged _best_result and extract_field_via_queries here)

# ---------- Worker ----------
from typing import Any, Tuple

# ---------- Helper: parse employee column from input ----------
def get_emp_num_from_input(emp_str: str) -> int | None:
    """
    Convert input string like '50-100', '1000+', '500' into an integer.
    - Range: take average
    - '1000+': treat as 10000
    - Single number: return as int
    """
    if not emp_str:
        return None
    emp_str = str(emp_str).strip().replace(",", "")
    # Handle range
    if re.search(rf"\d+{RANGE_DASH}\d+", emp_str):
        parts = re.split(RANGE_DASH, emp_str)
        try:
            nums = [int(p) for p in parts]
            return int(sum(nums) / len(nums))
        except ValueError:
            return None
    # Handle "1000+" type
    m = re.match(r"(\d+)\+", emp_str)
    if m:
        return int(m.group(1)) * 10  # 1000+ -> 10000
    # Handle single number
    try:
        return int(emp_str)
    except ValueError:
        return None



def find_company_info(
    domain: str,
    country: str,
    company: str,
    debug: bool = False,
    fields: list[str] | None = None,
    company_emp_input: dict | None = None,
) -> tuple[Any, Any, Any, bool, Any, Any]:
    """
    Fetch company information (employees, revenue, industry) using ZoomInfo + LinkedIn.

    Args:
        domain (str): Company domain
        country (str): Country
        company (str): Company name
        debug (bool): Enable debug logging
        fields (list[str] | None): Which fields to fetch. Options: 
            ["employees", "revenue", "industry"]. Default=None (all)
        company_emp_input (dict | None): Optional mapping of domain -> employee count from input CSV

    Returns:
        tuple: (emp, rev, ind, flagged, emp_num, rev_num)
    """
    if fields is None:
        fields = ["employees", "revenue", "industry"]

    emp, rev, ind = None, None, None
    emp_num, rev_num = None, None
    flagged = False

    try:
        if not domain and not company:
            if debug:
                logger.warning("find_company_info: No domain/company provided, skipping.")
            return emp, rev, ind, flagged, emp_num, rev_num

        domain_key = str(domain).lower().strip()

        # --- Cache hit check ---
        try:
            cache_hit = load_cache(domain_key, "all")
        except Exception:
            cache_hit = None
        if cache_hit:
            if debug:
                logger.debug(f"[CACHE] Using cached result for {domain_key}")
            # ensure tuple shape
            return tuple(cache_hit)

        # --- Employees ---
        if "employees" in fields:
            emp_candidates = []
            collected_results = []  # keep top snippets for LLM fallback
            for q in employee_queries(company, domain, country):
                results = search_zoominfo(q)
                if results:
                    # extend collected_results until we have enough snippets (keep small)
                    for r in results:
                        if len(collected_results) >= 6:
                            break
                        collected_results.append(r)
                for r in results:
                    href, title, body = r.get("href", ""), r.get("title", ""), r.get("body", "")
                    cand = extract_employees(f"{title} {body}")
                    if cand:
                        score = score_candidate(company, domain, title, body, href)
                        emp_candidates.append((score, cand))

            if emp_candidates:
                best_emp = max(emp_candidates, key=lambda x: x[0])
                emp = format_employee_value(best_emp[1])
                emp_num = parse_employees(best_emp[1])
                if debug:
                    logger.debug(f"[{domain}] Selected Employees: {emp} (Score={best_emp[0]})")

            # Fallback using LLM (fast; use only top 2 snippets)
            if not emp:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        emp_llm = semantic_extract(llm_text, "employees")
                        if emp_llm:
                            emp = emp_llm.strip()
                            emp_num = parse_employees(emp) if emp else None
                            if debug and emp:
                                logger.debug(f"[{domain}] LLM extracted employees: {emp}")
                except Exception as e:
                    if debug:
                        logger.debug(f"[{domain}] LLM employees fallback failed: {e}")

        # --- If revenue is requested but emp_num not found, use company_emp_input ---
        if "revenue" in fields and not emp_num and company_emp_input:
            emp_input_val = company_emp_input.get(domain_key, None)
            if emp_input_val:
                emp_num = get_emp_num_from_input(emp_input_val)
                emp = str(emp_num) if emp_num else None
                if debug:
                    logger.debug(f"[{domain}] Employees taken from input CSV for revenue: {emp}")

        # --- Revenue ---
        if "revenue" in fields:
            rev_candidates = []
            collected_results = []
            for q in revenue_queries(company, domain, country):
                results = search_zoominfo(q)
                if results:
                    for r in results:
                        if len(collected_results) >= 6:
                            break
                        collected_results.append(r)
                for r in results:
                    href, title, body = r.get("href", ""), r.get("title", ""), r.get("body", "")
                    cand = extract_revenue(f"{title} {body}", emp_num)
                    if cand:
                        score = score_candidate(company, domain, title, body, href)
                        rev_candidates.append((score, cand))

            if rev_candidates:
                best_rev = max(rev_candidates, key=lambda x: x[0])
                rev = format_revenue_value(best_rev[1])
                rev_num = parse_revenue(best_rev[1])
                if not sanity_check(emp_num, rev_num):
                    if debug:
                        logger.debug(f"[{domain}] Revenue sanity check failed. Ignoring revenue.")
                    rev, rev_num = None, None
                elif debug:
                    logger.debug(f"[{domain}] Selected Revenue: {rev} (Score={best_rev[0]})")

            # LLM fallback for revenue
            if not rev:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        rev_llm = semantic_extract(llm_text, "revenue")
                        if rev_llm:
                            rev = rev_llm.strip()
                            rev_num = parse_revenue(rev) if rev else None
                            if debug and rev:
                                logger.debug(f"[{domain}] LLM extracted revenue: {rev}")
                except Exception as e:
                    if debug:
                        logger.debug(f"[{domain}] LLM revenue fallback failed: {e}")

        # --- Industry ---
        if "industry" in fields:
            ind_candidates = []
            collected_results = []
            for q in industry_queries(company, domain, country):
                results = search_zoominfo(q)
                if results:
                    for r in results:
                        if len(collected_results) >= 6:
                            break
                        collected_results.append(r)
                for r in results:
                    href, title, body = r.get("href", ""), r.get("title", ""), r.get("body", "")
                    if "linkedin.com/company" not in href:
                        continue
                    cand = fetch_industry_from_linkedin_about(href)
                    if cand:
                        score = score_candidate(company, domain, title, body, href)
                        ind_candidates.append((score, cand))

            if ind_candidates:
                best_ind = max(ind_candidates, key=lambda x: x[0])
                ind = parse_industry_value(best_ind[1])
                if debug:
                    logger.debug(f"[{domain}] Selected Industry: {ind}")

            # LLM fallback for industry
            if not ind:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        ind_llm = semantic_extract(llm_text, "industry")
                        if ind_llm:
                            ind = ind_llm.strip()
                            if debug and ind:
                                logger.debug(f"[{domain}] LLM extracted industry: {ind}")
                except Exception as e:
                    if debug:
                        logger.debug(f"[{domain}] LLM industry fallback failed: {e}")

        # --- Flagging ---
        if "employees" in fields and "revenue" in fields:
            if not is_valid_rpe(emp_num, rev_num):
                flagged = True

        # --- Cache result ---
        try:
            save_cache(domain_key, "all", [emp, rev, ind, flagged, emp_num, rev_num])
        except Exception:
            # don't fail the whole function if cache save fails
            pass

    except Exception as e:
        logger.exception("find_company_info failed for %s: %s", domain, e)

    return emp, rev, ind, flagged, emp_num, rev_num


# ---------- Async Orchestration ----------
async def process_unique_domain(
    loop,
    executor,
    domain: str,
    country: str = "",
    company: str = "",
    debug: bool = False,
    fields: list[str] | None = None,
    company_emp_input: dict | None = None,  # NEW
):
    """
    Run find_company_info in a thread executor for a single domain.

    Args:
        loop: Asyncio event loop
        executor: ThreadPoolExecutor
        domain (str): Company domain
        country (str): Country
        company (str): Company name
        debug (bool): Enable debug logging
        fields (list[str] | None): Fields to fetch ["employees", "revenue", "industry"]
        company_emp_input (dict | None): Optional mapping of domain -> employee count from input CSV

    Returns:
        tuple: (emp, rev, ind, flagged, emp_num, rev_num)
    """
    try:
        domain = (domain or "").strip()
        country = (country or "").strip()
        company = (company or "").strip()

        if not domain:
            return (None, None, None, False, None, None)

        # Ensure fields has a default
        if fields is None:
            fields = ["employees", "revenue", "industry"]

        # --- Run find_company_info in executor ---
        result = await loop.run_in_executor(
            executor,
            lambda d=domain, c=country, n=company: find_company_info(
                domain=d,
                country=c,
                company=n,
                debug=debug,
                fields=fields,
                company_emp_input=company_emp_input
            ),
        )

        if not isinstance(result, tuple) or len(result) != 6:
            return (None, None, None, False, None, None)

        return result

    except Exception as e:
        logger.warning("Domain worker failed for %s: %s", domain, e, exc_info=True)
        return (None, None, None, False, None, None)


async def irvalue_logic(df: pd.DataFrame, debug: bool = False, fields: list[str] | None = None) -> pd.DataFrame:
    """
    Enrich dataframe with employee, revenue, and industry info per unique domain.
    Uses async + ThreadPoolExecutor for parallel domain lookups.

    Args:
        df (pd.DataFrame): Input dataframe with columns "Domain", "Country", "Company Name"
        debug (bool): Enable debug logging
        fields (list[str] | None): Which fields to fetch. Options: ["employees", "revenue", "industry"]. Default=None (all)

    Returns:
        pd.DataFrame: Enriched dataframe
    """
    df = df.copy()

    # --- Create mapping domain -> input employee string ---
    company_emp_input = {}
    if "Company Employees" in df.columns:
        company_emp_input = {
            str(d).lower(): emp
            for d, emp in zip(df["Domain"], df["Company Employees"])
            if str(d).strip()
        }

    # --- Ensure fields has a default value ---
    if fields is None:
        fields = ["employees", "revenue", "industry"]

    # --- Ensure required columns exist ---
    for col in ["Domain", "Country", "Company Name"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].astype(str).str.strip()

    # --- Setup Revenue_per_Employee ranges if present ---
    if "Revenue_per_Employee" in df.columns:
        try:
            set_rpe_range_from_data(df["Revenue_per_Employee"])
        except Exception as e:
            logger.warning("Failed to set RPE ranges: %s", e)

    # --- Build unique domain jobs ---
    domain_jobs = {
        d.lower(): (d, c, n)
        for d, c, n in zip(df["Domain"], df["Country"], df["Company Name"])
        if str(d).strip()
    }

    loop = asyncio.get_running_loop()
    results_by_domain = {}

    # --- Run jobs concurrently ---
    max_workers = min(12, os.cpu_count() or 6)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {
            key: asyncio.create_task(
                process_unique_domain(loop, executor, dom, country, company, debug, fields, company_emp_input)
            )
            for key, (dom, country, company) in domain_jobs.items()
        }

        for key, task in tqdm(tasks.items(), total=len(tasks), desc="IRValue Progress"):
            try:
                emp, rev, ind, flagged, emp_num, rev_num = await task
            except Exception as e:
                logger.exception("Task for %s raised: %s", key, e)
                emp, rev, ind, flagged, emp_num, rev_num = (None, None, None, False, None, None)
            results_by_domain[key] = (emp, rev, ind, flagged, emp_num, rev_num)

    # --- Helper to safely get results ---
    def _get_for_domain(domain: str):
        return results_by_domain.get(str(domain).lower(), (None, None, None, False, None, None))

    # --- Vectorized assignment ---
    df["discovered_employees_raw"] = df["Domain"].map(lambda d: _get_for_domain(d)[0])
    df["discovered_revenue_raw"] = df["Domain"].map(lambda d: _get_for_domain(d)[1])
    df["discovered_industry"] = df["Domain"].map(lambda d: _get_for_domain(d)[2])
    df["flagged_rpe"] = df["Domain"].map(lambda d: _get_for_domain(d)[3])
    df["discovered_employees"] = df["Domain"].map(lambda d: _get_for_domain(d)[4])
    df["discovered_revenue"] = df["Domain"].map(lambda d: _get_for_domain(d)[5])

    logger.info("IRValue: completed enrichment for %d unique domains", len(results_by_domain))
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

    result_df = asyncio.run(irvalue_logic(df, debug=args.debug))
    result_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"IRValue processing complete. Output saved to {args.output}")  # Removed emoji for Windows-safe output

if __name__ == "__main__":
    run_cli()
