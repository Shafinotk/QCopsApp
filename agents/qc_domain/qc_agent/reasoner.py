from typing import Optional

def make_reason(
    prefix: str, 
    provider: str, 
    evidence_url: Optional[str], 
    discovered_company: str = ""
) -> str:
    """
    Generate a reason string based on provider, evidence, and discovered company validation.
    
    Args:
        prefix (str): The explanation prefix.
        provider (str): Source/provider name.
        evidence_url (Optional[str]): Evidence link if available.
        discovered_company (str, optional): Status of the discovered company 
                                            ("Invalid domain", 
                                             "Company name not found",
                                             or actual company name).
    
    Returns:
        str: Formatted reason string.
    """
    # Handle invalid or missing company cases
    if discovered_company == "Invalid domain":
        return "Provided domain is invalid or unreachable."
    if discovered_company == "Company name not found":
        return "Domain reachable but company name could not be extracted."
    
    # Build the reason string
    parts = [prefix]
    if provider:
        parts.append(f"(source: {provider})")
    if evidence_url:
        parts.append(f"Evidence: {evidence_url}")
    
    return " ".join(parts)
