import os
import asyncio
import pandas as pd
from typing import Optional
from tqdm.asyncio import tqdm_asyncio

from .models import RowResult
from .searcher import get_searcher, Searcher, check_company_relation
from .fetcher import normalize_domain_cell, discover_domain, get_company_from_domain
from .matcher import classify_match, company_name_similarity as company_similarity
from .reasoner import make_reason
from .io_utils import load_cache, save_cache


async def process_row_async(
    row: pd.Series,
    company_col: str,
    domain_col: Optional[str],
    searcher: Searcher,
    provider: str,
    cache: dict,
    cache_path: str,
    limit: int,
    timeout: int
) -> RowResult:
    company = str(row.get(company_col, "")).strip()
    provided_domain = normalize_domain_cell(row.get(domain_col)) if domain_col else None

    ck = f"{company}||{provided_domain or ''}||{provider}"
    cached = cache.get(ck)
    if cached:
        return RowResult(**cached)

    discovered_dom = evidence_url = title = page_text = None

    # Retry logic to fetch domain and homepage
    for attempt in range(3):
        try:
            discovered_dom, evidence_url, title, page_text = await asyncio.to_thread(
                discover_domain,
                company,
                provided_domain,
                timeout + attempt * 5
            )
            if discovered_dom or evidence_url:
                break
        except Exception as e:
            print(f"[WARN] Error fetching domain for {company}: {e}")
        if attempt < 2:
            await asyncio.sleep(2 ** attempt)

    discovered_company, name_score, valid_name = get_company_from_domain(
        company,
        evidence_url or discovered_dom or provided_domain or ""
    )

    sim = None
    if discovered_company and company:
        sim = company_similarity(company, discovered_company)
        # Discard very low similarity (<40%) only if domain does NOT match
        if sim < 0.4 and not (provided_domain and discovered_dom and provided_domain.lower() == discovered_dom.lower()):
            discovered_dom = None
            discovered_company = None
            evidence_url = None

    if not (discovered_dom or provided_domain):
        rr = RowResult(
            provided_company=company,
            provided_domain=provided_domain,
            discovered_domain=None,
            discovered_company="Invalid domain",
            match_status="invalid_domain",
            reason="No valid homepage or domain found after retries.",
            evidence_url=None
        )
        cache[ck] = rr.__dict__
        save_cache(cache_path, cache)
        return rr

    # Classify match first
    status, reason_prefix = classify_match(
        provided_domain,
        discovered_dom,
        title,
        page_text,
        company,
        discovered_company
    )

    # --- Borderline similarity: 40–69% ---
    if sim is not None and 0.4 <= sim < 0.7:
        # Only run relation check if borderline similarity
        related = await asyncio.to_thread(check_company_relation, company, discovered_company)
        if related:
            status = "related"
            reason_prefix = f"Similarity borderline ({sim*100:.0f}%), relation confirmed by evidence."
        else:
            status = "need_manual_check"
            reason_prefix = f"Similarity borderline ({sim*100:.0f}%), requires manual verification."

    # --- Last-resort mismatch: domains match but company names differ ---
    elif status == "mismatch" and discovered_company:
        if provided_domain and discovered_dom and provided_domain.lower() == discovered_dom.lower() and company.lower() != discovered_company.lower():
            related = await asyncio.to_thread(check_company_relation, company, discovered_company)
            if related:
                status = "related"
                reason_prefix = "Domain matches, company names differ. Relation confirmed by evidence."
            else:
                status = "need_manual_check"
                reason_prefix = "Domain matches, company names differ. Manual verification required."

    # --- Final last-resort relation check for remaining mismatches ---
    if status == "mismatch" and discovered_company:
        related = await asyncio.to_thread(check_company_relation, company, discovered_company)
        if related:
            status = "related"
            reason_prefix = "No direct match, but relation found by evidence. Marked as related."
        else:
            status = "mismatch"
            reason_prefix = "No match or relation found. Marked as mismatch."

    reason = make_reason(
        reason_prefix,
        provider or "ddg",
        evidence_url,
        discovered_company
    )

    rr = RowResult(
        provided_company=company,
        provided_domain=provided_domain,
        discovered_domain=discovered_dom or provided_domain,
        discovered_company=discovered_company,
        match_status=status,
        reason=reason,
        evidence_url=evidence_url
    )

    # Update cache
    cache[ck] = rr.__dict__
    save_cache(cache_path, cache)
    return rr


async def process_dataframe_async(
    df: pd.DataFrame,
    company_col: str,
    domain_col: Optional[str],
    provider: str = "ddg",
    limit: int = 20,
    timeout: int = 25,
    cache_path: str = "qc_cache.json",
) -> pd.DataFrame:
    cache = load_cache(cache_path)
    searcher = get_searcher(provider)

    tasks = [
        process_row_async(
            row,
            company_col,
            domain_col,
            searcher,
            provider,
            cache,
            cache_path,
            limit,
            timeout,
        )
        for _, row in df.iterrows()
    ]

    results = await tqdm_asyncio.gather(*tasks)
    res_df = pd.DataFrame([r.__dict__ for r in results])
    return res_df


def merge_with_original(df: pd.DataFrame, res_df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df.reset_index(drop=True), res_df.reset_index(drop=True)], axis=1)
