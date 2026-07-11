"""Groq-based news sentiment: batch all of a symbol's articles into one call."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from ..config import Config
from ..models import Direction, NewsArticle, NewsVerdict

try:
    import groq  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for tests/local runs
    groq = SimpleNamespace(Groq=None)

log = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "direction": {"type": "string", "enum": ["up", "down", "neutral"]},
        "score": {"type": "number"},
        "confidence": {"type": "number"},
        "key_drivers": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["direction", "score", "confidence", "key_drivers", "rationale"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are a financial news analyst. Given recent news headlines and snippets "
    "about a stock, judge the likely near-term (next 1-3 trading days) price "
    "direction implied by the news. score is -1.0 (strongly bearish) to 1.0 "
    "(strongly bullish). confidence is 0.0 (no signal) to 1.0 (very confident), "
    "reflecting how much the news actually bears on price, not how confident you "
    "are in your reasoning.\n\n"
    "Respond with a single JSON object only, no other text, matching exactly this shape:\n"
    '{"direction": "up" | "down" | "neutral", "score": <number -1.0..1.0>, '
    '"confidence": <number 0.0..1.0>, "key_drivers": [<string>, ...], "rationale": <string>}'
)


def _build_prompt(symbol: str, articles: list[NewsArticle]) -> str:
    lines = [f"Symbol: {symbol}", "", "Recent articles:"]
    for a in articles:
        lines.append(f"- [{a.source}, {a.published_ts.isoformat()}] {a.title}")
        if a.snippet:
            lines.append(f"  {a.snippet}")
    return "\n".join(lines)


def analyze_news(symbol: str, articles: list[NewsArticle], cfg: Config) -> NewsVerdict:
    if not articles:
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="no recent news found", article_count=0,
        )

    if not cfg.groq_api_key:
        log.warning("GROQ_API_KEY not set; skipping sentiment analysis for %s", symbol)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (no Groq API key)", article_count=len(articles),
        )

    global groq
    if getattr(groq, "Groq", None) is None:
        try:
            import groq as groq_module  # type: ignore
        except ImportError:
            log.warning("groq package not installed; skipping sentiment analysis for %s", symbol)
            return NewsVerdict(
                symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
                rationale="NEWS_UNAVAILABLE (groq missing)", article_count=len(articles),
            )
        groq = groq_module

    client = groq.Groq(api_key=cfg.groq_api_key)
    try:
        response = client.chat.completions.create(
            model=cfg.groq_model,
            temperature=0.2,
            max_tokens=1024,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _build_prompt(symbol, articles)},
            ],
        )
    except Exception as exc:
        log.warning("groq sentiment call failed for %s: %s", symbol, exc)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (API error)", article_count=len(articles),
        )

    content = getattr(response.choices[0].message, "content", "")
    if not content:
        log.warning("groq sentiment returned empty content for %s", symbol)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (empty response)", article_count=len(articles),
        )

    if isinstance(content, list):
        text = "".join(getattr(part, "text", "") for part in content)
    else:
        text = str(content)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("groq sentiment returned invalid JSON for %s: %r", symbol, text)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (bad JSON)", article_count=len(articles),
        )

    try:
        direction = Direction(str(data["direction"]).upper())
        score = max(-1.0, min(1.0, float(data["score"])))
        confidence = max(0.0, min(1.0, float(data["confidence"])))
    except (KeyError, ValueError, TypeError) as exc:
        log.warning("groq sentiment returned an unexpected shape for %s: %r (%s)", symbol, data, exc)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (unexpected shape)", article_count=len(articles),
        )

    return NewsVerdict(
        symbol=symbol,
        direction=direction,
        score=score,
        confidence=confidence,
        key_drivers=list(data.get("key_drivers", []) or []),
        rationale=str(data.get("rationale", "")),
        article_count=len(articles),
    )
