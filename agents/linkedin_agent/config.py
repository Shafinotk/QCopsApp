import os
from dotenv import load_dotenv

load_dotenv()

# SEARCH_ENGINE: "bing" or "serpapi"
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "bing").lower()

BING_API_KEY = os.getenv("BING_API_KEY", "").strip()
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
