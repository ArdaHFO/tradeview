"""Google News RSS fetch + dedupe for a single symbol."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import quote

import feedparser

from ..config import Config
from ..models import NewsArticle

log = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _is_duplicate(title: str, seen: list[str], threshold: float) -> bool:
    norm = _normalize_title(title)
    for other in seen:
        if SequenceMatcher(None, norm, other).ratio() >= threshold:
            return True
    return False


def fetch_articles(symbol: str, company_name: str | None, cfg: Config) -> list[NewsArticle]:
    """Fetch recent, deduplicated news articles for a symbol via Google News RSS."""
    query = f"{company_name or symbol} stock"
    url = _RSS_URL.format(query=quote(query))
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("google news fetch failed for %s: %s", symbol, exc)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.news_lookback_hours)
    articles: list[NewsArticle] = []
    seen_titles: list[str] = []

    for entry in feed.entries:
        published = _entry_published(entry)
        if published is None or published < cutoff:
            continue
        title = entry.get("title", "").strip()
        if not title or _is_duplicate(title, seen_titles, cfg.dedupe_similarity_threshold):
            continue
        seen_titles.append(_normalize_title(title))
        articles.append(NewsArticle(
            title=title,
            source=entry.get("source", {}).get("title", "") if isinstance(entry.get("source"), dict) else "",
            url=entry.get("link", ""),
            published_ts=published,
            snippet=entry.get("summary", "").strip(),
        ))
        if len(articles) >= cfg.max_articles_per_symbol:
            break

    log.info("news fetch: %s -> %d articles", symbol, len(articles))
    return articles


def _entry_published(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)
