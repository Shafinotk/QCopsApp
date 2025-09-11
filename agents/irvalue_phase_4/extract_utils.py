import re
import numpy as np

RANGE_DASH = r"[-\u2013\u2014]"

# ---------- Employee Extraction ----------
def extract_employees(text: str):
    if not text:
        return None
    patterns = [
        rf'\b\d{{1,3}}(?:,\d{{3}})*\+?\s*Employees?\b',
        rf'\bEmployees?\s*:\s*\d{{1,3}}(?:,\d{{3}})*\+?\b',
        rf'\b<\s*\d+\s*Employees?\b',
        rf'\bCompany\s*size\s*\d{{1,3}}(?:,\d{{3}})*\+?\s*Employees?\b',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def format_employee_value(s: str):
    return s.strip() if s else None

def parse_employees(s):
    if not s:
        return None
    if isinstance(s, (int, float)):
        return int(s)
    digits = re.sub(r"[^\d]", "", str(s))
    return int(digits) if digits else None

# ---------- Revenue Extraction ----------
def extract_revenue(text: str):
    if not text:
        return None
    money = r'\$\s*[\d,.]+'
    unit = r'(?:\s*(?:Million|Billion|Thousand|M|B|K))?'
    patterns = [
        rf'{money}\s*{unit}\+?',
        rf'<\s*{money}\s*{unit}',
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return None

def format_revenue_value(s: str):
    return s.strip() if s else None

def parse_revenue(s):
    """
    Parse strings like '$29.6 Billion', '$16.5M', '$500K', '$58'
    into integer revenue in USD. Auto-corrects if value looks too small.
    """
    if not s:
        return None
    if isinstance(s, (int, float)):
        return int(s)

    s = s.replace("$", "").replace(",", "").strip().lower()

    # detect multiplier
    mult = 1
    if "billion" in s or "b" in s:
        mult = 1_000_000_000
    elif "million" in s or "m" in s:
        mult = 1_000_000
    elif "thousand" in s or "k" in s:
        mult = 1_000

    digits = re.findall(r"[\d.]+", s)
    if not digits:
        return None
    val = float(digits[0]) * mult

    # --- Auto-correct: if revenue < 1M and no unit provided, assume Million ---
    if val < 1_000_000 and mult == 1:
        val = val * 1_000_000

    return int(val)

# ---------- Industry Extraction ----------
def extract_industry(text: str):
    if not text:
        return None
    norm = text.replace("•", " | ").replace("·", " | ").replace("‧", " | ").replace("∙", " | ")
    norm = re.sub(r'\s+\|\s+', ' | ', norm)
    patterns = [
        r'\bIndustry\s*:\s*([\wÀ-ÿ &/,-]+)',
        r'\bSector\s*:\s*([\wÀ-ÿ &/,-]+)',
        r'(?:^|\s\|\s)Industry\s*\|\s*([\wÀ-ÿ &/,-]{2,100})(?:\s\|\s|$)',
        r'(?:^|\s\|\s)Sector\s*\|\s*([\wÀ-ÿ &/,-]{2,100})(?:\s\|\s|$)',
        r'^[^|]+\|\s*([\wÀ-ÿ &/,-]{2,100})\s*\|\s*LinkedIn\b',
        r'^[^—]+\—\s*([\wÀ-ÿ &/,-]{2,100})\s*\—\s*LinkedIn\b',
        r'^[^-]+\-\s*([\wÀ-ÿ &/,-]{2,100})\s*\-\s*LinkedIn\b',
    ]
    for p in patterns:
        m = re.search(p, norm, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" -|,").strip()
    return None

def normalize_industry(s: str):
    if not s:
        return None
    s = s.strip()
    mapping = {
        "Servicios financieros": "Financial Services",
        "Fabricación": "Manufacturing",
        "Tecnología de la información y servicios": "Information Technology & Services",
        "Educación superior": "Higher Education",
        "Comercio minorista": "Retail",
        "Construcción": "Construction",
    }
    return mapping.get(s, s)

def parse_industry_value(s: str):
    if not s:
        return None
    s = s.strip()
    if s.lower() in {"linkedin", "home", "overview", "profile"}:
        return None
    return normalize_industry(s) if 2 <= len(s) <= 100 else None

# ---------- Revenue per Employee Validation ----------
# Fixed hard-coded range
RPE_MIN, RPE_MAX = 60_000, 9_000_000  

def set_rpe_range_from_data(series):
    """
    (Disabled dynamic percentile-based range. Using fixed 60k–9M instead.)
    """
    global RPE_MIN, RPE_MAX
    RPE_MIN, RPE_MAX = 60_000, 9_000_000

def is_valid_rpe(revenue: int, employees: int) -> bool:
    global RPE_MIN, RPE_MAX
    if not revenue or not employees or employees <= 0:
        return False
    rpe = revenue / employees
    return RPE_MIN <= rpe <= RPE_MAX
