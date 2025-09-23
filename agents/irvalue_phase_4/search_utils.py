# agents/irvalue_phase_4/search_utils.py
from ddgs import DDGS
import requests
from requests.exceptions import RequestException
import os
import time
import logging
import re

# Logging: prints to stdout/stderr so Render shows it
logger = logging.getLogger("irvalue.search_utils")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

DEFAULT_TIMEOUT = 10  # seconds
DEFAULT_RETRIES = 3
RETRY_BACKOFF = 1.5  # multiplier

def _requests_get(url, headers=None, timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_RETRIES):
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)
    attempt = 0
    while attempt < max_retries:
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except RequestException as e:
            attempt += 1
            wait = RETRY_BACKOFF ** attempt
            logger.warning("GET %s failed (attempt %d/%d): %s — retrying after %.1fs", url, attempt, max_retries, e, wait)
            time.sleep(wait)
    logger.error("GET %s failed after %d attempts", url, max_retries)
    return None

def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip() if text else ""

def _use_serpapi(query, max_results=10):
    # Optional SerpAPI fallback if you set SERPAPI_KEY in env
    key = os.getenv("SERPAPI_KEY")
    if not key:
        return None
    try:
        params = {
            "engine": "google",
            "q": query,
            "num": max_results,
            "api_key": key
        }
        # Use requests to call SerpAPI
        r = requests.get("https://serpapi.com/search", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("organic_results", []):
            href = item.get("link") or item.get("url")
            title = item.get("title") or ""
            body = item.get("snippet") or ""
            results.append({"href": href, "title": title, "body": body})
        return results
    except Exception as e:
        logger.warning("SerpAPI fallback failed: %s", e)
        return None

def search_zoominfo(query, max_results=10):
    """
    Use ddgs (DuckDuckGo Search) first. If that fails or returns empty,
    optionally try SerpAPI (if SERPAPI_KEY is provided).
    """
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, timelimit=10, safesearch='Off'):
                if len(results) >= max_results:
                    break
                # ddgs returns dict with 'id', 'title', 'body', 'href' typically
                href = r.get("link") or r.get("href") or r.get("url")
                title = r.get("title", "")
                body = r.get("body", "")
                results.append({"href": href, "title": title, "body": body})
    except Exception as e:
        logger.warning("DDGS search failed for query %r: %s", query, e)

    if not results:
        # try SerpAPI fallback
        serp = _use_serpapi(query, max_results=max_results)
        if serp:
            return serp

    return results

def fetch_html(url, timeout=DEFAULT_TIMEOUT):
    """
    Robust HTML fetch with headers, retries and logging.
    """
    if not url:
        return None
    # some URLs returned by ddgs may be relative; ensure http scheme
    if url.startswith("//"):
        url = "https:" + url
    try:
        html = _requests_get(url, timeout=timeout, max_retries=DEFAULT_RETRIES)
        return html
    except Exception as e:
        logger.warning("fetch_html exception for %s: %s", url, e)
        return None

# Example helper that parses industry from LinkedIn-like pages (kept from original)
def fetch_industry_from_linkedin_about(url):
    html = fetch_html(url)
    if not html:
        return None
    m = re.search(r'"industry":"([^"]+)"', html, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r'Industry\s*</dt>\s*<dd[^>]*>([^<]+)</dd>', html, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None
