# Domain–Company QC Agent

Automate QC to check whether a **company's domain** matches the **company name** in a CSV.
The agent performs a **web search**, finds the likely official website, compares with the provided domain, and
adds a **Match Status** and **Reason** column. It also produces a formatted **Excel** with highlighted mismatches.

## Features
- Upload a CSV and run checks via **Streamlit** UI or Command Line.
- Uses **DuckDuckGo** by default (no API key needed). Optionally use **SerpAPI** or **Bing Web Search**.
- Heuristic + fuzzy matching on: registered domain, page title, brand tokens, redirect, "About" page signals.
- Outputs both CSV and XLSX (with color highlights for matches/mismatches/relations).
- Simple **JSON cache** to avoid repeated lookups.
- Extensible “search provider” interface to plug more engines.

## Input expectations
Your input file should contain at least two columns (header names are auto-detected case-insensitively):
- `company`, `company_name`, or similar
- `domain`, `website`, `url`, or similar

## Quickstart
```bash
# 1) Create and activate a virtual env (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Configure search API keys
cp .env.example .env
# Edit .env and set SERPAPI_API_KEY or BING_SUBSCRIPTION_KEY if you want those providers

# 4a) Run as CLI
python main.py /path/to/input.csv -o results.csv --xlsx results.xlsx --provider ddg --limit 5

# 4b) Or run the Streamlit app (UI upload & download)
streamlit run app_streamlit.py
```

## Output columns
- `Provided_Company_Name`
- `Provided_Domain`
- `Discovered_Domain` (agent’s best guess)
- `Match_Status` — one of: `match`, `related`, `mismatch`, `no_website`
- `Reason` — brief human-readable explanation
- `Evidence_URL` — link used as evidence

The XLSX output highlights:
- **Green** rows → match
- **Yellow** rows → related or no_website
- **Red** rows → mismatch

> Note: CSV cannot store visual highlights; that’s why we also write an XLSX with formatting.

## Column detection
We auto-map typical headers. If your columns are unusual, the Streamlit UI lets you choose them.

## Rate limits & ethics
- Be polite: we fetch only a tiny number of pages per company, obey timeouts, and add a user-agent string.
- Respect the search engine and site ToS. Prefer official APIs (SerpAPI, Bing).

## Troubleshooting
- If some sites block requests, try switching provider (`--provider serpapi` or `--provider bing`) or increase `--timeout`.
- If you get false mismatches for brands owned by parent companies, consider raising the **related** threshold in `matcher.py` or adding a manual exceptions file.

## License
MIT
