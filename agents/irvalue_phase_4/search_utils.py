from ddgs import DDGS
import requests
from requests.exceptions import RequestException
import re

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")

def clean_text(text: str) -> str:
    return text.replace("\n", " ").strip() if text else ""

def search_zoominfo(query, max_results=10):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, backend="duckduckgo"):
                title = clean_text(r.get("title", ""))
                href = (r.get("href") or "")
                body = clean_text(r.get("body", ""))
                if "zoominfo.com" in href.lower():
                    results.append({"title": title, "body": body, "href": href})
    except Exception:
        pass
    return results

def search_linkedin(query, max_results=10):
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, backend="duckduckgo"):
                title = clean_text(r.get("title", ""))
                href = (r.get("href") or "")
                body = clean_text(r.get("body", ""))
                if "linkedin.com/company" in href.lower():
                    results.append({"title": title, "body": body, "href": href})
    except Exception:
        pass
    return results

def fetch_html(url: str, timeout: int = 10) -> str | None:
    if not url:
        return None
    try:
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-US,en;q=0.9"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and resp.text:
            return resp.text
    except RequestException:
        return None
    return None

def fetch_linkedin_industry(url: str) -> str | None:
    """
    Fetch industry from LinkedIn About page (forced English if available).
    """
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
