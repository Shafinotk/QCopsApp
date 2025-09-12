# agents/linkedin_agent/linkedin_finder.py
import re
from typing import List, Optional, Tuple, Dict
from rapidfuzz import fuzz
from search_engine import search_web

# ---- thresholds & weights (tunable) ----
MAX_RESULTS_PER_QUERY = 60
NAME_MIN_SCORE = 65
COMPANY_MIN_SCORE = 55
TITLE_MIN_SCORE = 50
DOMAIN_STRICT_BONUS = 30
LINKEDIN_IN_BONUS = 20
WEIGHT_NAME = 0.50
WEIGHT_COMPANY = 0.30
WEIGHT_TITLE = 0.20


def build_queries(first_name: str, last_name: str, company: str, title: str, domain: str, email: str) -> List[str]:
    """
    Build queries strictly based on user rules:
    1) company name + first + last + linkedin
    2) email + linkedin
    3) first + last + domain + linkedin
    4) first + last + title + domain + linkedin
    5) first + last + title + company name + linkedin
    """
    queries: List[str] = []
    name = f"{first_name} {last_name}".strip()

    if name and company:
        queries.append(f'"{company}" "{name}" linkedin')
    if email:
        queries.append(f'"{email}" linkedin')
    if name and domain:
        queries.append(f'"{name}" "{domain}" linkedin')
    if name and title and domain:
        queries.append(f'"{name}" "{title}" "{domain}" linkedin')
    if name and title and company:
        queries.append(f'"{name}" "{title}" "{company}" linkedin')

    # remove duplicates
    seen, final = set(), []
    for q in queries:
        if q not in seen:
            final.append(q)
            seen.add(q)
    return final


def _join_text(res: Dict[str, str]) -> str:
    return " ".join([res.get("title", ""), res.get("body", ""), res.get("href", "")]).strip().lower()


def _token_presence_check(name: str, text: str) -> bool:
    tokens = [t for t in re.split(r"\s+", name.lower()) if t]
    if not tokens:
        return False
    return all(tok in text for tok in tokens)


def score_result(res: Dict[str, str], name: str, company: str, title: str, domain: str) -> Tuple[int, Dict[str, float]]:
    text = _join_text(res)
    href = (res.get("href") or "").lower()

    name_score = fuzz.token_set_ratio(name, text) if name else 0
    company_score = fuzz.token_set_ratio(company, text) if company else 0
    title_score = fuzz.partial_ratio(title, text) if title else 0

    domain_present = bool(domain and (domain.lower() in href or domain.lower() in text))

    composite = (WEIGHT_NAME * name_score) + (WEIGHT_COMPANY * company_score) + (WEIGHT_TITLE * title_score)
    if domain_present:
        composite += DOMAIN_STRICT_BONUS
    if "linkedin.com/in/" in href:
        composite += LINKEDIN_IN_BONUS

    breakdown = {
        "name_score": name_score,
        "company_score": company_score,
        "title_score": title_score,
        "domain_present": 1.0 if domain_present else 0.0,
        "composite_raw": composite
    }
    return int(round(composite)), breakdown


def _is_acceptable_candidate(res: Dict[str, str], name: str, company: str, title: str, domain: str) -> bool:
    text = _join_text(res)
    href = (res.get("href") or "").lower()

    # Name must always match
    name_ok = _token_presence_check(name, text) or (fuzz.token_set_ratio(name, text) >= NAME_MIN_SCORE)
    if not name_ok:
        return False

    # LinkedIn links → require only name
    if "linkedin.com/in/" in href:
        return True

    # Non-LinkedIn → must be company domain + (company or title match)
    domain_ok = bool(domain and (domain.lower() in href))
    company_ok = (company and fuzz.token_set_ratio(company, text) >= COMPANY_MIN_SCORE)
    title_ok = (title and fuzz.partial_ratio(title, text) >= TITLE_MIN_SCORE)

    return bool(domain_ok and (company_ok or title_ok))


def find_link(first_name: str, last_name: str, company: str, title: str, domain: str, email: str = "") -> Optional[str]:
    name = f"{first_name} {last_name}".strip()
    queries = build_queries(first_name, last_name, company, title, domain, email)

    candidates: List[Tuple[int, Dict[str, str], Dict[str, float]]] = []

    for q in queries:
        print(f"[linkedin_finder] Searching: {q}")
        results = search_web(q, max_results=MAX_RESULTS_PER_QUERY)
        if not results:
            continue
        for res in results:
            score, breakdown = score_result(res, name, company, title, domain)
            candidates.append((score, res, breakdown))

    accepted = [(s, r, b) for (s, r, b) in candidates if _is_acceptable_candidate(r, name, company, title, domain)]

    # Separate LinkedIn vs company-domain
    linkedin_candidates = [(s, r, b) for (s, r, b) in accepted if "linkedin.com/in/" in (r.get("href") or "").lower()]
    domain_candidates = [(s, r, b) for (s, r, b) in accepted if domain and domain.lower() in (r.get("href") or "").lower()]

    # Pick best LinkedIn first
    if linkedin_candidates:
        linkedin_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_res, best_breakdown = linkedin_candidates[0]
        print(f"[linkedin_finder] Best LinkedIn candidate: {best_res.get('href')} (score={best_score}) breakdown={best_breakdown}")
        return best_res.get("href")

    # Else fallback to best domain candidate
    if domain_candidates:
        domain_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_res, best_breakdown = domain_candidates[0]
        print(f"[linkedin_finder] Best domain candidate: {best_res.get('href')} (score={best_score}) breakdown={best_breakdown}")
        return best_res.get("href")

    print(f"[linkedin_finder] No suitable link found for {name} @ {company}/{domain}")
    return None
