import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import chardet
import logging
<<<<<<< HEAD
from tqdm import tqdm
from typing import Optional
=======
import os, json, hashlib
from transformers import pipeline
from pathlib import Path

>>>>>>> origin/main

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

<<<<<<< HEAD
# =========================================================
# 🧠 LOGGING
# =========================================================
=======

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
>>>>>>> origin/main
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

        # must mention company or domain
        if not (_company_matches(company_name, snippet) or (domain and domain.lower() in snippet.lower())):
            continue

        score = snippet_similarity_score(company_name, snippet)

        # ✅ ZoomInfo domain boost
        if "zoominfo" in href.lower():
            score += 1.0

        # ✅ relevant field keyword boost
        if any(k in snippet.lower() for k in field_keywords):
            score += 0.7

        # ✅ presence of both revenue & employees lowers confusion
        if "revenue" in snippet.lower() and "employees" in snippet.lower():
            score += 0.3

        # ✅ penalize overly noisy snippets (too many numbers)
        if len(re.findall(r"\d+", snippet)) > 3:
            score -= 0.5

        # ✅ major boost if extractor returns a value
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

        # ✅ Enforce domain filter
        if extractor.__name__ in ["extract_employees", "extract_revenue"]:
            if "zoominfo.com" not in href.lower():
                continue
        elif extractor.__name__ == "extract_industry":
            if "linkedin.com/company" not in href.lower():
                continue

        # ========================================================
        # Get snippet + score
        snippet, score = _score_result(r, company_name, domain, extractor)
        if not snippet:
            continue

        # ========================================================
        # Enforce company name check
        if enforce_company_match and not _company_matches(company_name, snippet):
            continue

        # ========================================================
        # Extract and parse values
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

# Load reference file once globally or pass path
REFERENCE_FILE_PATH = r"C:\desktop\thinkogent Auto\apps\qcopsappNEW2\qcopsappnew\agents\irvalue_phase_4\IR Values master file.xlsx"
reference_df = load_reference_file(REFERENCE_FILE_PATH)


# =========================================================
# ⚙️ COMPANY INFO AGGREGATOR
# =========================================================
def find_company_info(domain, country, company):
    # -----------------------
    # 1️⃣ Try file first
    # -----------------------
    emp_val, rev_val, ind_val = get_values_from_file(domain, reference_df)

<<<<<<< HEAD
    # -----------------------
    # 2️⃣ If missing, fall back to web searches
    # -----------------------
    if not emp_val:
        emp_val, _, _ = extract_field_via_queries(
            employee_queries(company, domain, country),
            company, domain,
            extract_employees, format_employee_value,
            search_zoominfo
=======
def find_company_info(
    domain: str,
    country: str,
    company: str,
    debug: bool = False,
    fields: list[str] | None = None,
    company_emp_input: dict | None = None,
) -> tuple[Any, Any, Any, bool, Any, Any, Any, Any, Any]:
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
        tuple: (
            emp, rev, ind, flagged, emp_num, rev_num,
            emp_score, rev_score, ind_score
        )
    """
    if fields is None:
        fields = ["employees", "revenue", "industry"]

    emp, rev, ind = None, None, None
    emp_num, rev_num = None, None
    emp_score, rev_score, ind_score = None, None, None
    flagged = False

    try:
        if not domain and not company:
            if debug:
                logger.warning("find_company_info: No domain/company provided, skipping.")
            return emp, rev, ind, flagged, emp_num, rev_num, emp_score, rev_score, ind_score

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
            # backward compatibility for older cache versions
            if len(cache_hit) == 6:
                return tuple(cache_hit) + (None, None, None)
            return tuple(cache_hit)

        # --- Employees ---
        if "employees" in fields:
            emp_candidates = []
            collected_results = []  # keep top snippets for LLM fallback
            for q in employee_queries(company, domain, country):
                results = search_zoominfo(q)
                if results:
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
                emp_score = best_emp[0]
                emp = format_employee_value(best_emp[1])
                emp_num = parse_employees(best_emp[1])
                if debug:
                    logger.debug(f"[{domain}] Selected Employees: {emp} (Score={emp_score})")

            # Fallback using LLM (fast; use only top 2 snippets)
            if not emp:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        emp_llm = semantic_extract(llm_text, "employees")
                        if emp_llm:
                            emp = emp_llm.strip()
                            emp_num = parse_employees(emp) if emp else None
                            emp_score = 0  # indicate LLM fallback (no score)
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
                rev_score = best_rev[0]
                rev = format_revenue_value(best_rev[1])
                rev_num = parse_revenue(best_rev[1])
                if not sanity_check(emp_num, rev_num):
                    if debug:
                        logger.debug(f"[{domain}] Revenue sanity check failed. Ignoring revenue.")
                    rev, rev_num, rev_score = None, None, None
                elif debug:
                    logger.debug(f"[{domain}] Selected Revenue: {rev} (Score={rev_score})")

            # LLM fallback for revenue
            if not rev:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        rev_llm = semantic_extract(llm_text, "revenue")
                        if rev_llm:
                            rev = rev_llm.strip()
                            rev_num = parse_revenue(rev) if rev else None
                            rev_score = 0  # LLM fallback (no score)
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
                ind_score = best_ind[0]
                ind = parse_industry_value(best_ind[1])
                if debug:
                    logger.debug(f"[{domain}] Selected Industry: {ind} (Score={ind_score})")

            # LLM fallback for industry
            if not ind:
                try:
                    llm_text = " ".join([r.get("title", "") + " " + r.get("body", "") for r in collected_results[:2]])
                    if llm_text.strip():
                        ind_llm = semantic_extract(llm_text, "industry")
                        if ind_llm:
                            ind = ind_llm.strip()
                            ind_score = 0  # LLM fallback
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
            save_cache(domain_key, "all", [
                emp, rev, ind, flagged, emp_num, rev_num,
                emp_score, rev_score, ind_score
            ])
        except Exception:
            pass

    except Exception as e:
        logger.exception("find_company_info failed for %s: %s", domain, e)

    return emp, rev, ind, flagged, emp_num, rev_num, emp_score, rev_score, ind_score



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
>>>>>>> origin/main
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

<<<<<<< HEAD
    # ThreadPool for async execution
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = {}
        for key, (dom, country, company) in jobs.items():
            # First try to get values from file
            emp_val, rev_val, ind_val = get_values_from_file(dom, reference_df)
=======
    # --- Run jobs concurrently ---
    max_workers = min(12, os.cpu_count() or 6)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = {
            key: asyncio.create_task(
                process_unique_domain(loop, executor, dom, country, company, debug, fields, company_emp_input)
            )
            for key, (dom, country, company) in domain_jobs.items()
        }
>>>>>>> origin/main

            # If any value is missing, fallback to web search
            if not emp_val or not rev_val or not ind_val:
                from functools import partial
                task = loop.run_in_executor(executor, partial(find_company_info, dom, country, company))
                tasks[key] = task
            else:
                # If all values found in file, wrap them in a completed future
                fut = asyncio.Future()
                # flagged RPE calculation
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

        # Await tasks
        for key, task in tqdm(tasks.items(), total=len(tasks), desc="IRValue Progress"):
            try:
<<<<<<< HEAD
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
=======
                # Await task and ensure tuple format
                result = await task
                if not isinstance(result, tuple) or len(result) != 6:
                    logger.warning(f"[{key}] Invalid result shape from task, using None placeholders")
                    result = (None, None, None, False, None, None)
                results_by_domain[key] = result
                if debug:
                    logger.debug(f"[CACHE/RESULT] Stored results for {key}: {result}")
            except Exception as e:
                logger.exception("Task for %s raised: %s", key, e)
                results_by_domain[key] = (None, None, None, False, None, None)

    # --- Map results back to dataframe ---
    df["Employees"], df["Revenue"], df["Industry"] = None, None, None
    df["Flagged"], df["Employee_Num"], df["Revenue_Num"] = False, None, None
    df["Cache_Status"] = "fresh"

    for idx, row in df.iterrows():
        key = str(row["Domain"]).lower().strip()
        if key in results_by_domain:
            emp, rev, ind, flagged, emp_num, rev_num = results_by_domain[key]
            df.at[idx, "Employees"] = emp
            df.at[idx, "Revenue"] = rev
            df.at[idx, "Industry"] = ind
            df.at[idx, "Flagged"] = flagged
            df.at[idx, "Employee_Num"] = emp_num
            df.at[idx, "Revenue_Num"] = rev_num
            # Optional: mark cached entries
            df.at[idx, "Cache_Status"] = "cached" if emp or rev or ind else "fresh"

    return df



# ---------- CLI Entry ----------
>>>>>>> origin/main
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
