from typing import Tuple, Optional, List
from rapidfuzz import fuzz
from .fetcher import registered_domain
from .searcher import check_company_relation

# thresholds
PROVIDED_MIN_MATCH = 50        # company vs domain similarity
DISCOVERED_STRONG = 70         # strong company-to-company match
BORDERLINE_MIN = 50            # borderline lower bound for relation check
DOMAIN_STRONG_MATCH = 70       # ✅ changed from 80 → 70 for domain-company similarity

BLACKLISTED_DOMAINS = {"facebook", "linkedin", "twitter", "instagram", "youtube"}


def tokenize_company(name: str) -> List[str]:
    import re
    name = (name or "").lower()
    name = re.sub(r"[^a-z0-9]+", " ", name)
    tokens = [t for t in name.split() if t]
    return tokens


def domain_root(domain: Optional[str]) -> str:
    if not domain:
        return ""
    dom = registered_domain(domain) or ""
    if not dom or "." not in dom:
        return ""  # invalid domain
    root = dom.split(".")[0].lower()
    return root


def company_domain_similarity(company: str, domain: Optional[str]) -> int:
    comp = " ".join(tokenize_company(company))
    dom = domain_root(domain)
    if not comp or not dom:
        return 0
    return int(fuzz.token_set_ratio(comp, dom))


def company_name_similarity(provided_company: str, discovered_company: Optional[str]) -> int:
    if not provided_company or not discovered_company:
        return 0

    pc = "".join(tokenize_company(provided_company))   # remove spaces
    dc = "".join(tokenize_company(discovered_company)) # remove spaces

    score1 = fuzz.token_set_ratio(pc, dc)
    score2 = fuzz.partial_ratio(pc, dc)
    score3 = fuzz.token_sort_ratio(pc, dc)

    return max(score1, score2, score3)


def classify_match(
    provided_domain: Optional[str],
    discovered_domain: Optional[str],
    title: str,
    page_text: str,
    company: str,
    discovered_company: Optional[str] = None,
) -> Tuple[str, str]:

    dd_root = domain_root(discovered_domain)
    pd_root = domain_root(provided_domain)

    # 🚫 Invalid domain → skip mapping
    if not pd_root or not dd_root:
        return "invalid_domain", f"Invalid or missing domain for company '{company}'."

    # --- Strong domain evidence overrides weak company-name similarity ---
    if pd_root == dd_root:
        sim_pd = company_domain_similarity(company, provided_domain)
        if sim_pd >= DOMAIN_STRONG_MATCH:
            return "match", (
                f"Domain root '{pd_root}' matches and company-domain similarity is strong ({sim_pd}%)."
            )

    # --- Company name similarity check ---
    if discovered_company:
        sim_company = company_name_similarity(company, discovered_company)

        # ✅ Strong match → accept
        if sim_company >= DISCOVERED_STRONG:
            return "match", (
                f"Provided company '{company}' strongly matches discovered company '{discovered_company}' "
                f"(similarity={sim_company}%)."
            )

        # ✅ Borderline (50–69) → require relation evidence
        if BORDERLINE_MIN <= sim_company < DISCOVERED_STRONG:
            if check_company_relation(company, discovered_company, max_results=10):
                return "related", (
                    f"Provided company '{company}' has borderline similarity ({sim_company}%) "
                    f"with discovered company '{discovered_company}', relation confirmed by evidence."
                )
            else:
                return "mismatch", (
                    f"Provided company '{company}' has borderline similarity ({sim_company}%) "
                    f"with discovered company '{discovered_company}', but no relation found."
                )

        # 🚫 Very low similarity → always mismatch
        if sim_company < BORDERLINE_MIN:
            return "mismatch", (
                f"Provided company '{company}' and discovered company '{discovered_company}' "
                f"have very low similarity ({sim_company}%), no relation check performed."
            )

    # --- Domain root alignment fallback ---
    if pd_root == dd_root and pd_root != "":
        sim_pd = company_domain_similarity(company, provided_domain)

        # Company name mismatch but domain matches
        if discovered_company and company_name_similarity(company, discovered_company) < DISCOVERED_STRONG:
            if company_name_similarity(company, discovered_company) >= BORDERLINE_MIN:
                if check_company_relation(company, discovered_company, max_results=10):
                    return "related", (
                        f"Domain root '{pd_root}' matches, but company names differ (similarity "
                        f"{company_name_similarity(company, discovered_company)}%). Relation confirmed."
                    )
                else:
                    return "need_manual_check", (
                        f"Domain root '{pd_root}' matches, but company names differ "
                        f"and no relation found. Manual check required."
                    )
            return "mismatch", (
                f"Domain root '{pd_root}' matches, but company names differ significantly."
            )

        if sim_pd >= PROVIDED_MIN_MATCH:
            return "match", (
                f"Exact match: Provided domain root '{pd_root}' = Discovered domain root '{dd_root}'."
            )

    # --- Fallback mismatch ---
    return "mismatch", (
        f"Domain '{pd_root}' and discovered '{dd_root}' do not align with company '{company}'."
    )
