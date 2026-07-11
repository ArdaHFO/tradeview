import json
from datetime import datetime, timezone
from types import SimpleNamespace

from forecaster.config import Config
from forecaster.models import Direction, NewsArticle
from forecaster.news import sentiment


def _cfg() -> Config:
    return Config(anthropic_api_key="fake-key")


def _article() -> NewsArticle:
    return NewsArticle(title="Apple beats earnings", source="Reuters", url="http://x",
                        published_ts=datetime.now(timezone.utc), snippet="Strong iPhone sales")


def _fake_response(stop_reason: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )


def test_no_articles_returns_neutral_zero_confidence():
    verdict = sentiment.analyze_news("AAPL", [], _cfg())
    assert verdict.direction == Direction.NEUTRAL
    assert verdict.confidence == 0.0
    assert verdict.article_count == 0


def test_valid_json_response_parsed(monkeypatch):
    payload = json.dumps({
        "direction": "up", "score": 0.6, "confidence": 0.8,
        "key_drivers": ["strong earnings"], "rationale": "beat expectations",
    })

    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(
                create=lambda **kw: _fake_response("end_turn", payload)
            )

    monkeypatch.setattr(sentiment.anthropic, "Anthropic", FakeClient)
    verdict = sentiment.analyze_news("AAPL", [_article()], _cfg())
    assert verdict.direction == Direction.UP
    assert verdict.score == 0.6
    assert verdict.confidence == 0.8
    assert verdict.key_drivers == ["strong earnings"]


def test_refusal_returns_unavailable(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(
                create=lambda **kw: _fake_response("refusal", "")
            )

    monkeypatch.setattr(sentiment.anthropic, "Anthropic", FakeClient)
    verdict = sentiment.analyze_news("AAPL", [_article()], _cfg())
    assert verdict.direction == Direction.NEUTRAL
    assert verdict.confidence == 0.0
    assert "NEWS_UNAVAILABLE" in verdict.rationale


def test_malformed_json_returns_unavailable(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.messages = SimpleNamespace(
                create=lambda **kw: _fake_response("end_turn", "{not valid json")
            )

    monkeypatch.setattr(sentiment.anthropic, "Anthropic", FakeClient)
    verdict = sentiment.analyze_news("AAPL", [_article()], _cfg())
    assert verdict.direction == Direction.NEUTRAL
    assert "NEWS_UNAVAILABLE" in verdict.rationale
