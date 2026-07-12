"""News fetch + dedupe for a single symbol, market-aware, multi-source.

Sources:
  * google   – Google News RSS (keyless). Always queried in the local-market
               locale *and* a global-English pass, so both local and
               international coverage surfaces.
  * yahoo    – Yahoo Finance headline RSS (keyless), keyed on the ticker.
  * finnhub  – Finnhub company-news JSON API (needs FINNHUB_API_KEY).
  * newsapi  – NewsAPI.org "everything" JSON API (needs NEWSAPI_KEY).

The single most important thing this module does is turn a *ticker* into a
query that matches how outlets actually write about the company. Searching
"THYAO.IS" returns almost nothing; "Türk Hava Yolları" (or "Turkish Airlines")
returns its full coverage. See build_news_query.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from ..config import Config
from ..models import NewsArticle

log = logging.getLogger(__name__)

_RSS_URL = "https://news.google.com/rss/search?q={query}&hl={hl}&gl={gl}&ceid={ceid}"
_YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region={gl}&lang={hl}"
_FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
_NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Order matters: this is also the order sources are queried/merged in.
AVAILABLE_SOURCES = ("google", "yahoo", "finnhub", "newsapi")

# Human-readable labels + whether the source needs an API key, for the UI.
SOURCE_LABELS: dict[str, str] = {
    "google": "Google Haberler",
    "yahoo": "Yahoo Finans",
    "finnhub": "Finnhub",
    "newsapi": "NewsAPI",
}
_KEYED_SOURCES = ("finnhub", "newsapi")

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
    r"anonim\s+sirketi|anonim\s+şirketi|anonim\s+ortakligi|anonim\s+ortaklığı|a\.?ş\.?|a\.?s\.?|a\.?o\.?|"
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


def _ticker_root(symbol: str) -> str:
    """AAPL -> AAPL, THYAO.IS -> THYAO — the exchange suffix isn't how news
    outlets refer to the company."""
    return symbol.split(".", 1)[0].strip() if symbol else ""


def _looks_like_ticker(name: str, symbol: str) -> bool:
    """True when the 'name' handed to us is really just the ticker.

    The detail view passes the symbol itself in the name slot when it has no
    company name, so "THYAO.IS"/"THYAO" as a name must not be treated as a
    company name — otherwise we'd build a ticker-only query again.
    """
    if not name:
        return True
    squashed = re.sub(r"[^a-z0-9]", "", name.lower())
    return squashed in {
        re.sub(r"[^a-z0-9]", "", symbol.lower()),
        _ticker_root(symbol).lower(),
    }


def build_news_query(symbol: str, company_name: str | None) -> str:
    """Build the news search query for a symbol.

    Prefers the company's real name (how outlets write about it) and OR's in
    the bare ticker root so ticker-tagged finance pieces still surface. Never
    appends a language-specific word like "stock" — that silently filtered out
    almost all non-English coverage (e.g. "Türk Hava Yolları stock" matched a
    handful of articles where "Türk Hava Yolları" alone matched hundreds).
    """
    root = _ticker_root(symbol)
    name = "" if _looks_like_ticker(company_name or "", symbol) else _strip_corporate_suffix(company_name or "")
    if not name:
        return root or symbol
    phrase = f'"{name}"' if " " in name else name
    if root and len(root) >= 3 and root.isalpha() and root.lower() != name.lower():
        return f"{phrase} OR {root}"
    return phrase


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


def _consider_article(articles: list[NewsArticle], *, title: str, source_label: str, url: str,
                      published: datetime | None, snippet: str, cfg: Config,
                      cutoff: datetime, seen_titles: list[str]) -> bool:
    """Add one article if it's recent enough and not a near-duplicate.

    Returns True once `articles` has reached max_articles_per_symbol, so callers
    can stop early. Shared by every source so dedupe/cutoff behave identically.
    """
    if published is None or published < cutoff:
        return False
    title = (title or "").strip()
    if not title or _is_duplicate(title, seen_titles, cfg.dedupe_similarity_threshold):
        return False
    seen_titles.append(_normalize_title(title))
    articles.append(NewsArticle(
        title=title,
        source=source_label,
        url=url or "",
        published_ts=published,
        snippet=(snippet or "").strip(),
    ))
    return len(articles) >= cfg.max_articles_per_symbol


def _parse_feed(feed, source_label: str, cfg: Config, cutoff: datetime, seen_titles: list[str]) -> list[NewsArticle]:
    articles: list[NewsArticle] = []
    for entry in feed.entries:
        full = _consider_article(
            articles,
            title=entry.get("title", ""),
            source_label=source_label,
            url=entry.get("link", ""),
            published=_entry_published(entry),
            snippet=entry.get("summary", ""),
            cfg=cfg, cutoff=cutoff, seen_titles=seen_titles,
        )
        if full:
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
    query = build_news_query(symbol, company_name)
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


def _http_get_json(url: str, timeout: float = 8.0) -> object | None:
    try:
        req = Request(url, headers={"User-Agent": "trd-news/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        log.warning("http json fetch failed for %s: %s", url.split("?", 1)[0], exc)
        return None


def _fetch_finnhub_articles(symbol: str, cfg: Config, seen_titles: list[str],
                             lookback_hours: int) -> list[NewsArticle]:
    if not cfg.finnhub_api_key:
        log.info("finnhub source skipped for %s: FINNHUB_API_KEY not set", symbol)
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    params = urlencode({
        "symbol": _ticker_root(symbol),   # Finnhub keys on the bare ticker
        "from": cutoff.date().isoformat(),
        "to": datetime.now(timezone.utc).date().isoformat(),
        "token": cfg.finnhub_api_key,
    })
    data = _http_get_json(f"{_FINNHUB_URL}?{params}")
    if not isinstance(data, list):
        return []
    articles: list[NewsArticle] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ts = item.get("datetime")
        published = datetime.fromtimestamp(ts, tz=timezone.utc) if isinstance(ts, (int, float)) and ts else None
        full = _consider_article(
            articles,
            title=item.get("headline", ""),
            source_label=f"Finnhub · {item.get('source', '')}".strip(" ·"),
            url=item.get("url", ""),
            published=published,
            snippet=item.get("summary", ""),
            cfg=cfg, cutoff=cutoff, seen_titles=seen_titles,
        )
        if full:
            break
    return articles


def _fetch_newsapi_articles(symbol: str, company_name: str | None, cfg: Config,
                             seen_titles: list[str], lookback_hours: int) -> list[NewsArticle]:
    if not cfg.newsapi_key:
        log.info("newsapi source skipped for %s: NEWSAPI_KEY not set", symbol)
        return []
    hl, _, _ = _locale_for_symbol(symbol)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    params = urlencode({
        "q": build_news_query(symbol, company_name),
        "language": hl.split("-", 1)[0],   # tr-TR -> tr
        "sortBy": "publishedAt",
        "pageSize": min(50, max(10, cfg.max_articles_per_symbol * 2)),
        "from": cutoff.date().isoformat(),
        "apiKey": cfg.newsapi_key,
    })
    data = _http_get_json(f"{_NEWSAPI_URL}?{params}")
    if not isinstance(data, dict) or data.get("status") != "ok":
        return []
    articles: list[NewsArticle] = []
    for item in data.get("articles", []):
        if not isinstance(item, dict):
            continue
        published = _parse_iso8601(item.get("publishedAt"))
        source = item.get("source") or {}
        full = _consider_article(
            articles,
            title=item.get("title", ""),
            source_label=f"NewsAPI · {source.get('name', '')}".strip(" ·"),
            url=item.get("url", ""),
            published=published,
            snippet=item.get("description", ""),
            cfg=cfg, cutoff=cutoff, seen_titles=seen_titles,
        )
        if full:
            break
    return articles


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


def available_sources(cfg: Config) -> list[dict]:
    """Describe every source for the UI: id, label, and whether it's usable
    right now (keyed sources are only usable once their API key is set)."""
    out: list[dict] = []
    for src in AVAILABLE_SOURCES:
        needs_key = src in _KEYED_SOURCES
        if src == "finnhub":
            available = bool(cfg.finnhub_api_key)
        elif src == "newsapi":
            available = bool(cfg.newsapi_key)
        else:
            available = True
        out.append({
            "id": src,
            "label": SOURCE_LABELS.get(src, src),
            "needs_key": needs_key,
            "available": available,
        })
    return out


def fetch_articles(symbol: str, company_name: str | None, cfg: Config,
                    sources: list[str] | None = None) -> list[NewsArticle]:
    """Fetch recent, deduplicated news articles for a symbol from one or more sources.

    Retries with a wider lookback window if the configured one turns up
    nothing — low-newsflow and internationally-listed stocks don't always
    have news in the last 24-72h even when older coverage exists.
    """
    selected_sources = _normalize_sources(sources)

    for step in _WIDEN_STEPS_HOURS:
        lookback_hours = _effective_lookback_hours(cfg) if step is None else step
        seen_titles: list[str] = []
        articles: list[NewsArticle] = []
        for source in selected_sources:
            if source == "google":
                articles.extend(_fetch_google_articles(symbol, company_name, cfg, seen_titles, lookback_hours))
            elif source == "yahoo":
                articles.extend(_fetch_yahoo_articles(symbol, cfg, seen_titles, lookback_hours))
            elif source == "finnhub":
                articles.extend(_fetch_finnhub_articles(symbol, cfg, seen_titles, lookback_hours))
            elif source == "newsapi":
                articles.extend(_fetch_newsapi_articles(symbol, company_name, cfg, seen_titles, lookback_hours))

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


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
