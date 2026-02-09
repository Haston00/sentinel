"""
SENTINEL — SEC EDGAR filings monitor.
Watches for significant filings (10-K, 10-Q, 8-K, insider trades).
"""
from __future__ import annotations

import pandas as pd
import requests

from utils.helpers import cache_key, is_cache_fresh, load_parquet, save_parquet
from utils.logger import get_logger

log = get_logger("data.sec_edgar")

SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
SEC_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"
SEC_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"

HEADERS = {"User-Agent": "Sentinel Market Intel research@sentinel.local"}

IMPORTANT_FORMS = ["10-K", "10-Q", "8-K", "4", "SC 13D", "SC 13G"]


def search_filings(
    query: str = "",
    forms: list[str] | None = None,
    date_range: str = "30d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Search recent SEC filings via EDGAR full-text search.
    """
    forms = forms or IMPORTANT_FORMS
    path = cache_key("sec", f"search_{query[:20]}_{date_range}")

    if use_cache and is_cache_fresh(path, max_age_hours=6):
        log.info("Cache hit: SEC filings")
        return load_parquet(path)

    log.info(f"Searching SEC EDGAR: {query or 'all'} (forms: {forms})")
    try:
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": query,
            "dateRange": f"custom",
            "startdt": pd.Timestamp.now() - pd.Timedelta(date_range),
            "enddt": pd.Timestamp.now(),
            "forms": ",".join(forms),
        }

        resp = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params=params,
            headers=HEADERS,
            timeout=15,
        )

        if resp.status_code != 200:
            log.warning(f"SEC EDGAR returned status {resp.status_code}")
            return pd.DataFrame()

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])

        rows = []
        for hit in hits:
            source = hit.get("_source", {})
            rows.append({
                "Company": source.get("display_names", [""])[0] if source.get("display_names") else "",
                "Form": source.get("form_type", ""),
                "Filed": source.get("file_date", ""),
                "Description": source.get("display_date_filed", ""),
                "URL": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&filenum={source.get('file_num', '')}",
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            save_parquet(df, path)
        log.info(f"SEC: {len(df)} filings found")
        return df

    except Exception as e:
        log.error(f"SEC EDGAR search failed: {e}")
        return pd.DataFrame()


def get_company_filings(ticker: str, forms: list[str] | None = None) -> pd.DataFrame:
    """Get recent filings for a specific company ticker."""
    return search_filings(query=ticker, forms=forms)
