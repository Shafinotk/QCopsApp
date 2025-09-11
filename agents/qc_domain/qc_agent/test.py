from ddgs import DDGS

# Query
q = f"is there any relation between Blacksky and the domain blacksky.com"

# Run search with English + US region enforced
with DDGS() as ddgs:
    for r in ddgs.text(
        q, 
        max_results=10,   # get more results to increase chances
        backend="duckduckgo",
        safesearch="off", # you can also set 'moderate' or 'strict'
        region="wt-wt",   # world/English results
    ):
        # each r is a dict with keys like: 'title', 'body', 'href'
        print("TITLE:", r.get("title"))
        print("HREF :", r.get("href"))
        print("SNIP :", r.get("body"))
        print("-" * 60)
