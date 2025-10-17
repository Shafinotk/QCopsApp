from ddgs import DDGS

companyname = "Great Eastern Holdings"
domain = "greateasternlife.com"
country = "Singapore"

# ✅ Updated query to search on ZoomInfo
q = f'how much turn over does {domain} have according to site:zoominfo.com'

# ✅ Use DDGS to fetch up to 30 text results
with DDGS() as ddgs:
    results = ddgs.text(q, max_results=30, backend="duckduckgo")

    count = 0
    for r in results:
        count += 1
        print(f"RESULT #{count}")
        print("TITLE:", r.get("title"))
        print("HREF :", r.get("href"))
        print("SNIP :", r.get("body"))
        print("-" * 150)
