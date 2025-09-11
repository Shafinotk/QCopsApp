# agents/linkedin_agent/linkedin_finder.py
import re
from typing import List, Optional
from search_engine import search_web

def build_queries(first_name: str, last_name: str, company: str, title: str, domain: str) -> List[str]:
    """Build prioritized queries to increase chance of finding a LinkedIn or domain link."""
    name = (f"{first_name} {last_name}").strip()
    queries = []

    # Strong name + linkedin
    if name:
        queries.append(f'"{name}" site:linkedin.com')
        queries.append(f'"{name}" site:linkedin.com/in')

    # Name + company/title
    if name and company:
        queries.append(f'"{name}" "{company}" site:linkedin.com')
    if name and title:
        queries.append(f'"{name}" "{title}" site:linkedin.com')

    # name + domain
    if name and domain:
        queries.append(f'"{name}" {domain} site:linkedin.com')

    # company or domain pages if person-based queries fail
    if company:
        queries.append(f'"{company}" site:linkedin.com/company')
    if domain:
        queries.append(f'{domain} site:linkedin.com')

    # final fallback: name only (no site restriction)
    if name:
        queries.append(f'"{name}"')

    # ensure queries are unique while preserving order
    seen = set()
    out = []
    for q in queries:
        if q not in seen:
            out.append(q)
            seen.add(q)
    return out

def extract_valid_links(urls: List[str], domain: str) -> List[str]:
    """
    Return LinkedIn profile/company links if present, otherwise fallback to domain-based links.
    Normalizes and filters candidate URLs.
    """
    if not urls:
        return []

    # normalize and filter
    normalized = []
    for u in urls:
        try:
            s = u.strip()
            if s:
                normalized.append(s)
        except Exception:
            continue

    linkedin_links = [u for u in normalized if re.search(r"linkedin\.com/(in|pub|company)/", u, flags=re.I)]
    if linkedin_links:
        return linkedin_links

    # fallback: any link that contains the company domain
    if domain:
        domain_lower = domain.lower()
        domain_links = [u for u in normalized if domain_lower in u.lower()]
        if domain_links:
            return domain_links

    # last fallback: return first valid url-looking string
    for u in normalized:
        if re.match(r'^https?://', u, flags=re.I):
            return [u]

    return []

def find_link(first_name: str, last_name: str, company: str, title: str, domain: str) -> Optional[str]:
    """
    Synchronous search that returns the first matching LinkedIn or domain link, or None.
    Prints debug info for each query.
    """
    queries = build_queries(first_name or "", last_name or "", company or "", title or "", domain or "")
    for q in queries:
        print(f"[linkedin_finder] Searching: {q}")
        urls = search_web(q, max_results=10)
        if not urls:
            print(f"[linkedin_finder] No results for query: {q}")
            continue
        links = extract_valid_links(urls, domain or "")
        if links:
            print(f"[linkedin_finder] Found link(s) for query `{q}`: {links[:3]}")
            return links[0]
    print(f"[linkedin_finder] No link found for {first_name} {last_name} @ {company} / {domain}")
    return None
