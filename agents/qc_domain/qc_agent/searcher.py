import os
from typing import List, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from ddgs import DDGS

# Domains we want to avoid in search results
BLACKLISTED_DOMAINS = {
    "linkedin", "crunchbase", "wikipedia", "facebook", "instagram", "twitter", "x",
    "youtube", "glassdoor", "indeed", "g2", "trustpilot", "angel"
}

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


def extract_domain_root(url: str) -> str:
    try:
        hostname = urlparse(url).hostname or ""
        parts = hostname.split(".")
        if len(parts) >= 2:
            return parts[-2].lower()
        return hostname.lower()
    except Exception:
        return ""


def is_blacklisted(url: str) -> bool:
    root = extract_domain_root(url)
    return root in BLACKLISTED_DOMAINS


class Searcher:
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        raise NotImplementedError


class DuckDuckGoSearcher(Searcher):
    def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        results: List[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results * 2, region="in-en", safesearch="moderate"):
                url = r.get("href") or r.get("url")
                title = r.get("title") or ""
                snippet = r.get("body") or ""
                if url and not is_blacklisted(url):
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
                if len(results) >= max_results:
                    break
        return results


# --- Improved relation check ---
def check_company_relation(company1: str, company2: str, max_results: int = 10) -> bool:
    """
    Check if two companies have a relation using search snippets.
    Only returns True if both company names appear together and strong evidence keywords are found.
    """
    relation_keywords = [
        "acquired", "subsidiary", "parent company", "partnered with", "merger",
        "owned by", "rebrand", "formerly known as", "transitioned to",
        "part of", "brand of", "merged into", "division of"
    ]

    # Normalize names for matching
    company1_lower = company1.lower()
    company2_lower = company2.lower()

    queries = [
        f"{company1} {company2} acquisition OR parent OR subsidiary OR merger OR rebrand OR formerly",
        f"{company1} {company2} relation OR partnership OR ownership"
    ]

    for q in queries:
        with DDGS() as ddgs:
            for r in ddgs.text(q, max_results=max_results, region="wt-wt", safesearch="moderate"):
                title = (r.get("title") or "").lower()
                snippet = (r.get("body") or "").lower()
                url = (r.get("href") or "").lower()

                combined = " ".join([title, snippet, url])

                # ✅ Must contain both company names AND a relation keyword
                if company1_lower in combined and company2_lower in combined:
                    if any(k in combined for k in relation_keywords):
                        return True
    return False


def get_searcher(preferred: Optional[str] = None) -> Searcher:
    preferred = (preferred or "").lower()
    if preferred == "ddg":
        return DuckDuckGoSearcher()
    return DuckDuckGoSearcher()
