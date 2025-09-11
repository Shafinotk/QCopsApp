# agents/linkedin_agent/search_engine.py
from ddgs import DDGS
from typing import List

def search_web(query: str, max_results: int = 10) -> List[str]:
    """
    Search DuckDuckGo (via ddgs) and return a list of result URLs.
    Synchronous and defensive (prints errors to stdout).
    """
    results = []
    try:
        print(f"[search_web] Query: {query}")
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                # ddgs result dict often contains 'href' or 'url'
                if "href" in r and r["href"]:
                    results.append(r["href"])
                elif "url" in r and r["url"]:
                    results.append(r["url"])
    except Exception as e:
        # Print so wrapper / streamlit can show the problem
        print(f"[search_web] Error for query `{query}`: {e}")
    return results
