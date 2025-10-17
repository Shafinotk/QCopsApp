import re
import logging
import statistics

logger = logging.getLogger("irvalue.extract_utils")

# =========================================================
# 📊 IMPROVED FIELD EXTRACTION HELPERS
# =========================================================

def extract_employees(text: str):
    """
    Extracts clean employee-count expressions such as:
    '2,500 employees', 'between 1,000 and 5,000 employees', 'has 700 staff'
    """
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)

    # Capture explicit ranges first
    range_pattern = re.search(
        r"(?:between|from)\s*(\d{1,3}(?:,\d{3})*|\d+)\s*(?:and|to)\s*(\d{1,3}(?:,\d{3})*|\d+)\s*(?:employees?|staff|personnel|workforce)\b",
        text, re.I)
    if range_pattern:
        try:
            lo = int(range_pattern.group(1).replace(",", ""))
            hi = int(range_pattern.group(2).replace(",", ""))
            mid = int(statistics.median([lo, hi]))
            return f"{mid:,} employees"
        except Exception:
            pass

    # Standard explicit forms
    patterns = [
        r'(\d{1,3}(?:,\d{3})+|\d+)\s*(?:employees?|staff|personnel|workforce)\b',
        r'\b(?:employees?|staff|personnel|workforce)\s*[:\-]?\s*(\d{1,3}(?:,\d{3})+|\d+)\b',
        r'has\s+(\d{1,3}(?:,\d{3})+|\d+)\s+(?:employees?|staff)\b'
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            val = m.group(1)
            # reject absurd counts
            num = int(val.replace(",", ""))
            if 5 <= num <= 500000:
                return f"{num:,} employees"
    return None


def format_employee_value(s: str):
    return s.strip() if s else None


def extract_revenue(text: str):
    """
    Extracts revenue values with stricter money checks and ranges.
    """
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)

    # Range form e.g. 'between $10M and $20M'
    range_pattern = re.search(
        r"(?:between|from)\s*\$?\s?(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|m|k)\s*(?:and|to)\s*\$?\s?(\d+(?:\.\d+)?)\s*(billion|million|thousand|bn|m|k)",
        text, re.I)
    if range_pattern:
        lo = float(range_pattern.group(1))
        hi = float(range_pattern.group(3))
        unit = range_pattern.group(2).lower()
        mid = (lo + hi) / 2
        return f"${mid:.1f} {normalize_unit(unit)}"

    # Explicit money forms
    patterns = [
        r'\$\s?\d+(?:\.\d+)?\s?(?:billion|million|thousand|bn|m|k)\b',
        r'(?:less than|under|below|<)\s?\$?\s?\d+(?:\.\d+)?\s?(?:billion|million|thousand|bn|m|k)\b',
        r'(?:revenue|annual revenue|turnover)\s*(?:of|is|:)?\s*\$?\s?\d+(?:\.\d+)?\s?(?:billion|million|thousand|bn|m|k)\b'
    ]
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            val = re.sub(r'\s+', ' ', m.group(0)).strip()
            val = normalize_money(val)
            return val
    return None


def normalize_unit(u: str):
    u = u.lower()
    if u in ["b", "bn", "billion"]:
        return "Billion"
    if u in ["m", "mm", "million"]:
        return "Million"
    if u in ["k", "thousand"]:
        return "Thousand"
    return ""


def normalize_money(s: str):
    s = s.replace("B", "Billion").replace("M", "Million").replace("K", "Thousand")
    s = re.sub(r'\b(billion|million|thousand)\b', lambda m: m.group(1).capitalize(), s, flags=re.I)
    if not s.startswith("$"):
        s = "$" + s
    return s.strip()


def format_revenue_value(s: str):
    if not s:
        return None
    return normalize_money(s.strip())


def extract_industry(text: str):
    if not text:
        return None
    m = re.search(r'(?:industry|sector|field)\s*[:\-–]\s*([A-Za-z &/]+)', text, re.I)
    return m.group(1).strip() if m else None


def parse_industry_value(value: str):
    return value.strip().title() if value else None


# =========================================================
# ⚖️ VALIDATION UTILITIES
# =========================================================

def employees_to_number(value: str):
    if not value:
        return None
    try:
        num = float(re.sub(r"[^\d.]", "", value))
        return num if num > 0 else None
    except Exception:
        return None


def revenue_to_number(value: str):
    if not value:
        return None
    s = value.replace(",", "").replace("$", "").replace("USD", "").strip().upper()
    try:
        if "B" in s:
            return float(re.sub("[^0-9.]", "", s)) * 1_000_000_000
        if "M" in s:
            return float(re.sub("[^0-9.]", "", s)) * 1_000_000
        if "K" in s:
            return float(re.sub("[^0-9.]", "", s)) * 1_000
        return float(s)
    except Exception:
        return None


def is_valid_rpe(revenue: float, employees: float) -> bool:
    if not revenue or not employees or employees == 0:
        return True
    rpe = revenue / employees
    return 10_000 < rpe < 2_000_000


# =========================================================
# 🔍 SIMPLE TEXT SIMILARITY
# =========================================================

def snippet_similarity_score(company_name: str, snippet: str):
    if not snippet or not company_name:
        return 0
    snippet = snippet.lower()
    company_name = company_name.lower()
    overlap = sum(1 for w in company_name.split() if w in snippet)
    return overlap / max(len(company_name.split()), 1)
