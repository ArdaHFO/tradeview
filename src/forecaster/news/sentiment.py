"""Claude-based news sentiment: batch all of a symbol's articles into one call."""
from __future__ import annotations

import json
import logging

import anthropic

from ..config import Config
from ..models import Direction, NewsArticle, NewsVerdict

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
    "are in your reasoning."
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

    if not cfg.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set; skipping sentiment analysis for %s", symbol)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (no API key)", article_count=len(articles),
        )

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    try:
        response = client.messages.create(
            model=cfg.claude_model,
            max_tokens=1024,
            system=_SYSTEM,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": _SCHEMA}},
            messages=[{"role": "user", "content": _build_prompt(symbol, articles)}],
        )
    except (anthropic.APIError, anthropic.APIConnectionError) as exc:
        log.warning("claude sentiment call failed for %s: %s", symbol, exc)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (API error)", article_count=len(articles),
        )

    if response.stop_reason == "refusal":
        log.warning("claude sentiment refused for %s", symbol)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (refused)", article_count=len(articles),
        )

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("claude sentiment returned invalid JSON for %s: %r", symbol, text)
        return NewsVerdict(
            symbol=symbol, direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
            rationale="NEWS_UNAVAILABLE (bad JSON)", article_count=len(articles),
        )

    return NewsVerdict(
        symbol=symbol,
        direction=Direction(data["direction"].upper()),
        score=float(data["score"]),
        confidence=float(data["confidence"]),
        key_drivers=list(data.get("key_drivers", [])),
        rationale=data.get("rationale", ""),
        article_count=len(articles),
    )
