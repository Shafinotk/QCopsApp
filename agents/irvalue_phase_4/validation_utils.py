from rapidfuzz import fuzz
import logging
import tldextract

logger = logging.getLogger(__name__)

def validate_domain(url: str, target_domain: str) -> bool:
    """
    Validates if the URL's registered domain matches the target domain.
    Uses tldextract for robust domain matching.
    """
    if not url or not target_domain:
        return False

    extracted_url = tldextract.extract(url)
    extracted_target = tldextract.extract(target_domain)

    return (extracted_url.registered_domain.lower() ==
            extracted_target.registered_domain.lower())

def is_same_company(target_name: str, text: str, threshold: int = 80) -> bool:
    """Checks if text likely refers to the same company using fuzzy matching."""
    if not target_name or not text:
        return False
    return fuzz.partial_ratio(target_name.lower(), text.lower()) >= threshold

def sanity_check(employees: int | None, revenue: int | None) -> bool:
    """Simple rule-based sanity check for employee/revenue numbers."""
    if employees is None or revenue is None:
        return False
    if employees > 10000 and revenue < 1_000_000:  # too many employees but very low revenue
        return False
    if employees < 20 and revenue > 1_000_000_000:  # very few employees but huge revenue
        return False
    return True

def score_candidate(company: str, domain: str, title: str, body: str, href: str, debug: bool = False) -> int:
    """
    Score a search result candidate based on multiple relevance signals.
    Higher scores indicate better matches.
    """
    company = (company or "").strip()
    domain = (domain or "").strip()
    title = (title or "").strip()
    body = (body or "").strip()
    href = (href or "").strip().lower()

    score = 0
    breakdown = []

    # --- Domain match ---
    if validate_domain(href, domain):
        score += 50  # slightly higher weight to strongly favor same-domain results
        breakdown.append("domain_match=+50")
    elif domain:
        score -= 15  # stronger penalty for off-domain results
        breakdown.append("domain_mismatch=-15")

    # --- Fuzzy company name match (title) ---
    company_match_score = fuzz.partial_ratio(company.lower(), title.lower()) if company and title else 0
    score += company_match_score
    breakdown.append(f"fuzzy_title_match=+{company_match_score}")

    # penalize weak match
    if company_match_score < 60 and company:
        score -= 10
        breakdown.append("weak_company_match=-10")

    # --- Company mentions in body ---
    if company:
        mentions = body.lower().count(company.lower())
        if mentions > 0:
            bonus = min(mentions * 10, 30)  # cap at +30
            score += bonus
            breakdown.append(f"body_mentions={mentions} => +{bonus}")

    # --- LinkedIn priority ---
    if "linkedin.com/company" in href:
        score += 40
        breakdown.append("linkedin_bonus=+40")

    if debug:
        logger.debug(f"Score breakdown for {href or 'N/A'} -> {', '.join(breakdown)} | TOTAL={score}")

    return score
