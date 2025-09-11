from ddgs import DDGS


companyname = "KPMG"
domain = "kpmg.ca"
country = "Canada"


q = f'linkedin industry of {domain} {country} site:linkedin.com'

with DDGS() as ddgs:
    for r in ddgs.text(q, max_results=6, backend="duckduckgo"):
        # each r is a dict with keys like: 'title', 'body', 'href'
        print("TITLE:", r.get("title"))
        print("HREF :", r.get("href"))
        print("SNIP :", r.get("body"))
        print("-" * 60)

