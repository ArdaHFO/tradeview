from datetime import datetime, timezone

from forecaster import pipeline
from forecaster.config import Config
from forecaster.models import Bar, Direction, NewsVerdict, TechnicalVerdict


def _bars(close: float, count: int = 60) -> list[Bar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(ts=start, open=close, high=close + 1, low=close - 1, close=close, volume=1_000_000)
        for _ in range(count)
    ]


def test_compare_across_profiles_fetches_news_once_per_symbol(monkeypatch):
    news_calls: list[str] = []

    class DummyRecorder:
        def __init__(self, db_path):
            pass

        def record(self, prediction, user_id=None):
            pass

        def close(self):
            pass

    def fake_fetch_articles(symbol, company_name, cfg, sources=None):
        return []

    def fake_analyze_news(symbol, articles, cfg):
        news_calls.append(symbol)
        return NewsVerdict(symbol=symbol, direction=Direction.UP, score=0.4, confidence=0.5, rationale="news")

    def fake_fetch_bars(symbol, cfg, timeframe="1d"):
        return _bars(100.0)

    def fake_score_technical(symbol, bars):
        return TechnicalVerdict(symbol=symbol, score=0.2, reasons=["tech"])

    monkeypatch.setattr(pipeline, "PredictionRecorder", DummyRecorder)
    monkeypatch.setattr(pipeline, "fetch_articles", fake_fetch_articles)
    monkeypatch.setattr(pipeline, "analyze_news", fake_analyze_news)
    monkeypatch.setattr(pipeline, "fetch_bars", fake_fetch_bars)
    monkeypatch.setattr(pipeline, "score_technical", fake_score_technical)
    monkeypatch.setattr(pipeline.backfill, "run", lambda cfg, user_id=None: None)

    cfg = Config(groq_api_key="x")
    symbols = [
        {"symbol": "AAPL", "name": "Apple", "timeframe": "1d", "profile": p, "news_sources": ["google"]}
        for p in ("balanced", "news_heavy", "technical_heavy")
    ]
    results = pipeline.run_for_symbols(symbols, cfg)

    assert len(results) == 3
    assert news_calls == ["AAPL"]  # not fetched/analyzed 3x for 3 profiles
