from dataclasses import dataclass
from typing import Optional

@dataclass
class RowResult:
    provided_company: str
    provided_domain: Optional[str]
    discovered_domain: Optional[str]
    discovered_company: Optional[str] = ""   # NEW
    match_status: str = "mismatch"
    reason: str = ""
    evidence_url: str = ""

