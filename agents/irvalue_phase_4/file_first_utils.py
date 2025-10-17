# agents/irvalue_phase_4/file_first_utils.py

import pandas as pd
import re
from typing import Optional, Tuple

logger_name = "irvalue.file_first_utils"
import logging
logger = logging.getLogger(logger_name)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def load_reference_file(file_path: str) -> pd.DataFrame:
    """
    Load reference Excel/CSV and extract Employees, Industry, and Revenue 
    from the combined 'IR Values' column (e.g. '1,000 Employees,Software Development,$210.1 M').
    """
    try:
        # Load file
        if file_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path, dtype=str)
        else:
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)

        # Clean strings safely
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        # Ensure Domain column exists
        if "Domain" not in df.columns:
            logger.warning("Reference file missing 'Domain' column")
            df["Domain"] = ""

        # If IR Values exist, extract subfields
        if "IR Values" in df.columns:
            emp_list, rev_list, ind_list = [], [], []

            for val in df["IR Values"]:
                if not isinstance(val, str):
                    emp_list.append(None)
                    ind_list.append(None)
                    rev_list.append(None)
                    continue

                val = val.strip()

                # 🧠 Extract employees
                emp_match = re.search(r"[\d,]+\s*employees?", val, re.I)
                emp = emp_match.group(0).strip() if emp_match else None

                # 🧠 Extract revenue ($xxx M / B etc.)
                rev_match = re.search(r"\$\s?[\d,\.]+\s?(?:[MBK]|Million|Billion|Thousand)?", val, re.I)
                rev = rev_match.group(0).strip() if rev_match else None

                # 🧠 Extract industry (middle part)
                parts = [p.strip() for p in val.split(",") if p.strip()]
                ind = None
                if len(parts) >= 2:
                    for p in parts:
                        # Skip anything that's just numbers or ranges (e.g., 10, 2475, 501-1)
                        if re.fullmatch(r"[\d,\-\s]+", p):
                            continue
                        # Skip anything with employees or $
                        if re.search(r"(employee|\$)", p, re.I):
                            continue
                        # Skip empty or meaningless text
                        if not re.search(r"[A-Za-z]", p):
                            continue
                        ind = p.strip()
                        break
                    
                emp_list.append(emp)
                rev_list.append(rev)
                ind_list.append(ind)

            df["discovered_employees"] = emp_list
            df["discovered_industry"] = ind_list
            df["discovered_revenue"] = rev_list

        return df

    except Exception as e:
        logger.exception(f"Failed to load reference file: {e}")
        return pd.DataFrame()


def employee_in_range(employee_str: str, skip_ranges=[(500, 1000)]) -> bool:
    """
    Check if employee count is in any of the skip_ranges.
    Accepts formatted string like "752 employees" or "500-1000 employees".
    """
    if not employee_str:
        return False

    # Extract numbers
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d+", employee_str)]
    if not numbers:
        return False

    # Check for range or single number
    emp_value = numbers[0] if len(numbers) == 1 else sum(numbers)//2

    for lo, hi in skip_ranges:
        if lo <= emp_value <= hi:
            return True
    return False


def get_values_from_file(domain: str, reference_df: pd.DataFrame):
    if "Domain" not in reference_df.columns:
        logger.warning("Reference file missing 'Domain' column")
        return None, None, None

    row = reference_df[reference_df["Domain"].str.lower() == domain.lower()]
    if row.empty:
        return None, None, None

    emp = row["discovered_employees"].values[0] or None
    rev = row["discovered_revenue"].values[0] or None
    ind = row["discovered_industry"].values[0] or None

    # Skip employee if in range
    if emp and employee_in_range(emp):
        emp = None

    return emp, rev, ind
