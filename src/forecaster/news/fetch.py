"""Google News / Yahoo RSS fetch + dedupe for a single symbol, market-aware."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import quote

from ..config import Config
from ..models import NewsArticle

log = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={gl}&lang={hl}"

AVAILABLE_SOURCES = ("google", "yahoo")

# If nothing turns up in the configured lookback window, retry with these
# progressively wider windows before giving up. Low-newsflow and
# internationally-listed stocks often don't publish every single day —
# a single fixed 24h window was returning "no news" even when Google News
# had matches, just not inside the last day.
_WIDEN_STEPS_HOURS: tuple[int | None, ...] = (None, 168, 720)  # cfg default, 7d, 30d

# Exchange suffix -> (Google hl, gl, ceid) locale. Falls back to en-US/US for
# unsuffixed tickers (US exchanges) and anything not in this table.
_EXCHANGE_LOCALES: dict[str, tuple[str, str, str]] = {
    "IS": ("tr-TR", "TR", "TR:tr"),   # Istanbul (BIST)
    "PA": ("fr-FR", "FR", "FR:fr"),   # Paris (Euronext)
    "DE": ("de-DE", "DE", "DE:de"),   # Xetra
    "F": ("de-DE", "DE", "DE:de"),    # Frankfurt
    "L": ("en-GB", "GB", "GB:en"),    # London
    "MI": ("it-IT", "IT", "IT:it"),   # Milan
    "AS": ("nl-NL", "NL", "NL:nl"),   # Amsterdam
    "MC": ("es-ES", "ES", "ES:es"),   # Madrid
    "SW": ("de-CH", "CH", "CH:de"),   # Swiss
    "T": ("ja-JP", "JP", "JP:ja"),    # Tokyo
    "HK": ("zh-HK", "HK", "HK:zh"),   # Hong Kong
    "KS": ("ko-KR", "KR", "KR:ko"),   # Korea
    "SA": ("pt-BR", "BR", "BR:pt"),   # Sao Paulo (B3)
}
_DEFAULT_LOCALE = ("en-US", "US", "US:en")

# Common legal-entity suffixes across markets; stripping them makes the news
# query match how outlets actually refer to the company (e.g. "Aselsan", not
# "ASELSAN Elektronik Sanayi ve Ticaret Anonim Sirketi").
_CORPORATE_SUFFIX_RE = re.compile(
    r"[,\s]+(?:"
    r"anonim\s+sirketi|anonim\s+şirketi|anonim\s+ortakligi|anonim\s+ortaklığı|a\.?ş\.?|a\.?o\.?|"
    r"incorporated|inc\.?|corporation|corp\.?|company|co\.?|"
    r"limited|ltd\.?|plc|"
    r"s\.?p\.?a\.?|n\.?v\.?|a\.?g\.?|a\.?b\.?|s\.?a\.?|se"
    r")\.?\s*$",
    re.IGNORECASE,
)


def _strip_corporate_suffix(name: str) -> str:
    cleaned = name.strip()
    while True:
        next_cleaned = _CORPORATE_SUFFIX_RE.sub("", cleaned).strip()
        if next_cleaned == cleaned or not next_cleaned:
            break
        cleaned = next_cleaned
    return cleaned or name


def _locale_for_symbol(symbol: str) -> tuple[str, str, str]:
    if "." not in symbol:
        return _DEFAULT_LOCALE
    suffix = symbol.rsplit(".", 1)[-1].upper()
    return _EXCHANGE_LOCALES.get(suffix, _DEFAULT_LOCALE)


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def _is_duplicate(title: str, seen: list[str], threshold: float) -> bool:
    norm = _normalize_title(title)
    for other in seen:
        if SequenceMatcher(None, norm, other).ratio() >= threshold:
            return True
    return False


def _effective_lookback_hours(cfg: Config) -> int:
    """Bridge the Monday news gap: a fixed lookback misses Fri-evening/weekend
    news when a run happens Monday morning, since markets (and most news
    cadence) were quiet Sat/Sun.
    """
    if datetime.now(timezone.utc).weekday() == 0:  # Monday
        return max(cfg.news_lookback_hours, 72)
    return cfg.news_lookback_hours


def _parse_feed(feed, source_label: str, cfg: Config, cutoff: datetime, seen_titles: list[str]) -> list[NewsArticle]:
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


def _query_google_once(query: str, hl: str, gl: str, ceid: str, cfg: Config,
                        cutoff: datetime, seen_titles: list[str], symbol: str) -> list[NewsArticle]:
    import feedparser

    url = _RSS_URL.format(query=quote(query), hl=hl, gl=gl, ceid=ceid)
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("google news fetch failed for %s: %s", symbol, exc)
        return []
    return _parse_feed(feed, "Google News", cfg, cutoff, seen_titles)


def _fetch_google_articles(symbol: str, company_name: str | None, cfg: Config,
                            seen_titles: list[str], lookback_hours: int) -> list[NewsArticle]:
    hl, gl, ceid = _locale_for_symbol(symbol)
    query = f"{company_name or symbol} stock"
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)

    articles = _query_google_once(query, hl, gl, ceid, cfg, cutoff, seen_titles, symbol)

    # For internationally-listed companies, a large multinational is often
    # covered mostly by global English-language finance press, not local
    # outlets — restricting to the local market's gl/hl alone can miss most
    # of the coverage. Merge in a global query too (seen_titles dedupes).
    if (hl, gl, ceid) != _DEFAULT_LOCALE:
        articles += _query_google_once(query, *_DEFAULT_LOCALE, cfg, cutoff, seen_titles, symbol)

    return articles


def _fetch_yahoo_articles(symbol: str, cfg: Config, seen_titles: list[str],
                           lookback_hours: int) -> list[NewsArticle]:
    import feedparser

    hl, gl, _ = _locale_for_symbol(symbol)
    url = _YAHOO_RSS_URL.format(symbol=quote(symbol), hl=hl, gl=gl)
    try:
        feed = feedparser.parse(url)
    except Exception as exc:
        log.warning("yahoo news fetch failed for %s: %s", symbol, exc)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
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
    """Fetch recent, deduplicated news articles for a symbol from one or more RSS sources.

    Retries with a wider lookback window if the configured one turns up
    nothing — low-newsflow and internationally-listed stocks don't always
    have news in the last 24-72h even when older coverage exists.
    """
    selected_sources = _normalize_sources(sources)
    clean_name = _strip_corporate_suffix(company_name) if company_name else None

    for step in _WIDEN_STEPS_HOURS:
        lookback_hours = _effective_lookback_hours(cfg) if step is None else step
        seen_titles: list[str] = []
        articles: list[NewsArticle] = []
        for source in selected_sources:
            if source == "google":
                articles.extend(_fetch_google_articles(symbol, clean_name, cfg, seen_titles, lookback_hours))
            elif source == "yahoo":
                articles.extend(_fetch_yahoo_articles(symbol, cfg, seen_titles, lookback_hours))

        if articles:
            articles.sort(key=lambda article: article.published_ts, reverse=True)
            articles = articles[:cfg.max_articles_per_symbol]
            if step is not None:
                log.info("news fetch: %s found results only after widening lookback to %dh", symbol, step)
            log.info("news fetch: %s -> %d articles (lookback=%dh)", symbol, len(articles), lookback_hours)
            return articles

    log.info("news fetch: %s -> 0 articles even after widening to 30 days", symbol)
    return []


def _entry_published(entry) -> datetime | None:
    parsed = entry.get("published_parsed")
    if not parsed:
        return None
    return datetime(*parsed[:6], tzinfo=timezone.utc)
