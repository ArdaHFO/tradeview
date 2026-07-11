"""Google News RSS fetch + dedupe for a single symbol."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import quote

from ..config import Config
from ..models import NewsArticle

log = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

AVAILABLE_SOURCES = ("google", "yahoo")


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _is_duplicate(title: str, seen: list[str], threshold: float) -> bool:
    norm = _normalize_title(title)
    for other in seen:
        if SequenceMatcher(None, norm, other).ratio() >= threshold:
            return True
    return False


def _parse_feed(feed, source_label: str, cfg: Config, cutoff: datetime, seen_titles: list[str]) -> list[NewsArticle]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.news_lookback_hours)
    articles: list[NewsArticle] = []

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
            source=source_label,
            url=entry.get("link", ""),
            published_ts=published,
            snippet=entry.get("summary", "").strip(),
        ))
        if len(articles) >= cfg.max_articles_per_symbol:
            break

    return articles


def _fetch_google_articles(symbol: str, company_name: str | None, cfg: Config,
                          seen_titles: list[str]) -> list[NewsArticle]:
    import feedparser

    query = f"{company_name or symbol} stock"
    url = _RSS_URL.format(query=quote(query))
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("google news fetch failed for %s: %s", symbol, exc)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.news_lookback_hours)
    return _parse_feed(feed, "Google News", cfg, cutoff, seen_titles)


def _fetch_yahoo_articles(symbol: str, cfg: Config, seen_titles: list[str]) -> list[NewsArticle]:
    import feedparser

    url = _YAHOO_RSS_URL.format(symbol=quote(symbol))
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("yahoo news fetch failed for %s: %s", symbol, exc)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.news_lookback_hours)
    return _parse_feed(feed, "Yahoo Finance", cfg, cutoff, seen_titles)


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if not sources:
        return ["google"]
    normalized: list[str] = []
    for source in sources:
        label = source.strip().lower()
        if label == "all":
            return list(AVAILABLE_SOURCES)
        if label in AVAILABLE_SOURCES and label not in normalized:
            normalized.append(label)
    return normalized or ["google"]


def fetch_articles(symbol: str, company_name: str | None, cfg: Config,
                   sources: list[str] | None = None) -> list[NewsArticle]:
    """Fetch recent, deduplicated news articles for a symbol from one or more RSS sources."""
    selected_sources = _normalize_sources(sources)
    articles: list[NewsArticle] = []
    seen_titles: list[str] = []

    for source in selected_sources:
        if source == "google":
            articles.extend(_fetch_google_articles(symbol, company_name, cfg, seen_titles))
        elif source == "yahoo":
            articles.extend(_fetch_yahoo_articles(symbol, cfg, seen_titles))

    articles.sort(key=lambda article: article.published_ts, reverse=True)
    return articles[:cfg.max_articles_per_symbol]

    log.info("news fetch: %s -> %d articles", symbol, len(articles))
    return articles


def _entry_published(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)
