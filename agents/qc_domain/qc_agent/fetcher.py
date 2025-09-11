import requests, re, tldextract
from urllib.parse import urlparse, urlunparse
from bs4 import BeautifulSoup
from typing import Optional, Tuple
from rapidfuzz import fuzz
from .searcher import DuckDuckGoSearcher

UA = "Mozilla/5.0 (compatible; QC-Agent/1.0; +https://example.com/qa)"

BLACKLISTED_DOMAINS = {
    "google.com","support.google.com","youtube.com","wikipedia.org",
    "facebook.com","twitter.com","linkedin.com","instagram.com",
    "zhihu.com","reddit.com","bing.com","yahoo.com","ask.com",
    "baidu.com","duckduckgo.com","stackoverflow.com","quora.com",
    "github.com","crunchbase.com","glassdoor.com","indeed.com",
    "wordpress.com","blogspot.com","wixsite.com","weebly.com",
    "medium.com","substack.com","github.io","notion.site",
    "sites.google.com","zoominfo.com","dnb.com"
}

def normalize_url(url: str) -> Optional[str]:
    if not url:
        return None
    if not re.match(r"^https?://", url, re.I):
        url = "http://" + url.strip()
    try:
        p = urlparse(url)
        if not p.netloc:
            return None
        return urlunparse((p.scheme, p.netloc, p.path or "/", "", "", ""))
    except Exception:
        return None

def registered_domain(host_or_url: str) -> Optional[str]:
    if not host_or_url:
        return None
    try:
        host = urlparse(host_or_url).netloc or host_or_url
        ext = tldextract.extract(host)
        if not ext.domain:
            return None
        return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    except Exception:
        return None

def normalize_domain_cell(cell: Optional[str]) -> Optional[str]:
    if not cell or str(cell).strip() == "" or str(cell).lower() in {"nan","none"}:
        return None
    val = str(cell).strip().lower()
    dom = registered_domain(val)
    return dom or val

def fetch_page(url: str, timeout: int = 12) -> Tuple[str, str]:
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": UA}, allow_redirects=True)
        final_url = resp.url
        if resp.status_code >= 400:
            return final_url, ""
        return final_url, resp.text or ""
    except Exception:
        return url, ""

def extract_title_and_text(html: str) -> Tuple[str, str]:
    if not html:
        return "", ""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.text.strip() if soup.title else ""
    for tag in soup(["script","style","noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ").strip()
    text = re.sub(r"\s+", " ", text)
    return title[:300], text[:50000]

def get_company_from_domain(provided_company: str, domain: str, timeout: int = 10, threshold: int = 50):
    if not domain:
        return "Invalid domain", 0, False
    url = f"http://{domain}" if not domain.startswith(("http://","https://")) else domain
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            discovered_company = ""
            if soup.title and soup.title.string:
                discovered_company = soup.title.string.strip()
            elif soup.find("meta", property="og:site_name"):
                discovered_company = soup.find("meta", property="og:site_name").get("content", "").strip()
            elif soup.find("h1"):
                discovered_company = soup.find("h1").get_text(strip=True)

            if discovered_company:
                score = fuzz.partial_ratio(provided_company.lower(), discovered_company.lower())
                return discovered_company, score, score >= threshold
            else:
                return "Company name not found", 0, False
        else:
            return "Invalid domain", 0, False
    except Exception:
        return "Invalid domain", 0, False

def find_homepage_from_company(company: str, timeout: int = 10, similarity_threshold: int = 50) -> Optional[str]:
    """
    Search DuckDuckGo for the company's homepage, but only return URLs
    whose page title has reasonable similarity to the company name.
    """
    searcher = DuckDuckGoSearcher()
    results = searcher.search(f"{company} official site OR homepage", max_results=5)
    for r in results:
        dom = registered_domain(r.url)
        if not dom or dom in BLACKLISTED_DOMAINS:
            continue

        # Fetch and extract title before accepting
        final_url, html = fetch_page(r.url, timeout)
        if not html:
            continue
        title, _ = extract_title_and_text(html)
        if title:
            score = fuzz.partial_ratio(company.lower(), title.lower())
            if score >= similarity_threshold:
                return final_url  # ✅ Accept only if relevant enough
    return None

def discover_domain(company: str, provided_domain: Optional[str], timeout: int = 12) -> Tuple[str, str, str, str]:
    """
    Discover domain with validation:
    1. Try provided domain → validate homepage.
    2. If invalid, search for company homepage and validate by title similarity.
    Returns: (discovered_domain, homepage_url, title, page_text)
    """
    homepage_url = None
    discovered_dom = None
    html = ""

    # Step 1: Try provided domain first
    if provided_domain:
        homepage_url = f"http://{provided_domain}" if not provided_domain.startswith("http") else provided_domain
        homepage_url, html = fetch_page(homepage_url, timeout)
        discovered_dom = registered_domain(homepage_url)

    # Step 2: If no HTML OR domain invalid, search by company name
    if not html or not discovered_dom:
        found_homepage = find_homepage_from_company(company, timeout)
        if found_homepage:
            homepage_url, html = fetch_page(found_homepage, timeout)
            title, _ = extract_title_and_text(html)
            score = fuzz.partial_ratio(company.lower(), title.lower())
            if score >= 70:  # accept only strong match
                discovered_dom = registered_domain(homepage_url)
            else:
                homepage_url = html = discovered_dom = None  # reject if low similarity

    title, page_text = extract_title_and_text(html)
    return discovered_dom or "", homepage_url or "", title, page_text
