from rapidfuzz import fuzz

def validate_domain(url: str, target_domain: str) -> bool:
    if not url or not target_domain:
        return False
    return target_domain.lower() in url.lower()

def is_same_company(target_name: str, text: str, threshold: int = 80) -> bool:
    if not target_name or not text:
        return False
    return fuzz.partial_ratio(target_name.lower(), text.lower()) >= threshold

def sanity_check(employees: int | None, revenue: int | None) -> bool:
    if not employees or not revenue:
        return False
    if employees > 10000 and revenue < 1_000_000:
        return False
    if employees < 20 and revenue > 1_000_000_000:
        return False
    return True

# NEW: candidate scoring
def score_candidate(company: str, domain: str, title: str, body: str, href: str) -> int:
    """
    Score a search result candidate based on multiple signals.
    Higher is better.
    """
    score = 0

    # Domain match
    if validate_domain(href, domain):
        score += 40

    # Fuzzy company name match
    score += fuzz.partial_ratio(company.lower(), (title or "").lower())

    # Extra weight if company appears in body
    if company.lower() in (body or "").lower():
        score += 20

    # LinkedIn always strong for industry
    if "linkedin.com/company" in (href or "").lower():
        score += 50

    return score
