# agents/linkedin_agent/search_engine.py
from ddgs import DDGS
from typing import List, Dict

def search_web(query: str, max_results: int = 40) -> List[Dict[str, str]]:
    """
    Search DuckDuckGo (via ddgs) and return a list of result dicts with:
      - href: the url
      - title: result title
      - body: result snippet (if available)
    This lets callers validate against title/snippet, not only the URL.
    """
    results: List[Dict[str, str]] = []
    try:
        print(f"[search_web] Query: {query}")
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                href = r.get("href") or r.get("url") or ""
                title = r.get("title") or ""
                body = r.get("body") or ""
                if href:
                    results.append({"href": href, "title": title, "body": body})
    except Exception as e:
        print(f"[search_web] Error for query `{query}`: {e}")
    return results
