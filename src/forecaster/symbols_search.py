"""Symbol search for the UI dropdown, backed by yfinance's Yahoo Finance search."""
from __future__ import annotations

import logging

import yfinance as yf

log = logging.getLogger(__name__)


def search_symbols(query: str, limit: int = 8) -> list[dict]:
    """Search US-listed equities by ticker or company name."""
    query = query.strip()
    if not query:
        return []
    try:
        results = yf.Search(query, max_results=limit * 2).quotes
    except Exception as exc:
        log.warning("symbol search failed for %r: %s", query, exc)
        return []

    out: list[dict] = []
    seen: set[str] = set()
    for q in results:
        sym = q.get("symbol", "")
        if not sym or sym in seen or "." in sym or "-" in sym:
            continue  # skip dual-class/foreign-listed suffixes
        if q.get("quoteType") != "EQUITY":
            continue
        seen.add(sym)
        out.append({"symbol": sym, "name": q.get("longname") or q.get("shortname") or sym})
        if len(out) >= limit:
            break
    return out
