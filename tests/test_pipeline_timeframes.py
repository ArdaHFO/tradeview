from datetime import datetime, timezone
from types import SimpleNamespace

from forecaster.config import Config
from forecaster.models import Bar, Direction, NewsVerdict, TechnicalVerdict
from forecaster import pipeline


def _bars(close: float, count: int = 60) -> list[Bar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(ts=start, open=close, high=close + 1, low=close - 1, close=close, volume=1_000_000)
        for _ in range(count)
    ]


def test_run_for_symbols_keeps_same_symbol_timeframes_separate(monkeypatch):
    recorded = []

    class DummyRecorder:
        def __init__(self, db_path):
            pass
        def record(self, prediction):
            recorded.append((prediction.symbol, prediction.timeframe, prediction.price_at_prediction))
        def close(self):
            pass

    def fake_fetch_articles(symbol, company_name, cfg, sources=None):
        return []

    def fake_analyze_news(symbol, articles, cfg):
        return NewsVerdict(symbol=symbol, direction=Direction.UP, score=0.4, confidence=0.5, rationale="news")

    def fake_fetch_bars(symbol, cfg, timeframe="1d"):
        price = {"1d": 100.0, "1h": 200.0}.get(timeframe, 150.0)
        return _bars(price)

    def fake_score_technical(symbol, bars):
        return TechnicalVerdict(symbol=symbol, score=0.2, reasons=["tech"])

    monkeypatch.setattr(pipeline, "PredictionRecorder", DummyRecorder)
    monkeypatch.setattr(pipeline, "fetch_articles", fake_fetch_articles)
    monkeypatch.setattr(pipeline, "analyze_news", fake_analyze_news)
    monkeypatch.setattr(pipeline, "fetch_bars", fake_fetch_bars)
    monkeypatch.setattr(pipeline, "score_technical", fake_score_technical)
    monkeypatch.setattr(pipeline.backfill, "run", lambda cfg: None)

    cfg = Config(groq_api_key="x")
    results = pipeline.run_for_symbols([
        {"symbol": "AAPL", "name": "Apple", "timeframe": "1d", "profile": "balanced", "news_sources": ["google"]},
        {"symbol": "AAPL", "name": "Apple", "timeframe": "1h", "profile": "balanced", "news_sources": ["google"]},
    ], cfg)

    assert len(results) == 2
    assert recorded == [("AAPL", "1d", 100.0), ("AAPL", "1h", 200.0)]