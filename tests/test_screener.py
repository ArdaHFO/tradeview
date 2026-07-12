from datetime import datetime, timezone

from forecaster import screener
from forecaster.config import Config
from forecaster.models import Bar


def _bars(trend: float, count: int = 60) -> list[Bar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(ts=start, open=100 + i * trend, high=100 + i * trend + 1,
            low=100 + i * trend - 1, close=100 + i * trend, volume=1_000_000)
        for i in range(count)
    ]


def test_list_universes_shape():
    universes = screener.list_universes()
    ids = {u["id"] for u in universes}
    assert {"bist", "us", "eu"} <= ids
    for u in universes:
        assert u["count"] > 0 and u["label"]


def test_signal_label_thresholds():
    assert screener._signal_label(0.8) == "Güçlü Al"
    assert screener._signal_label(0.2) == "Al"
    assert screener._signal_label(0.0) == "Nötr"
    assert screener._signal_label(-0.2) == "Sat"
    assert screener._signal_label(-0.8) == "Güçlü Sat"


def test_scan_ranks_by_score_desc(monkeypatch):
    # Uptrending symbols should score above downtrending ones and the result
    # must come back sorted strongest-first.
    def fake_fetch_bars(symbol, cfg, timeframe="1d"):
        return _bars(1.0 if symbol.startswith(("A", "M", "N")) else -1.0)

    monkeypatch.setattr(screener, "fetch_bars", fake_fetch_bars)
    rows = screener.scan("us", Config(groq_api_key=""), "1d")

    assert rows, "expected scan results"
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert rows[0]["score"] >= rows[-1]["score"]
    assert set(rows[0]) == {"symbol", "name", "score", "direction", "signal", "price", "rsi"}


def test_scan_skips_symbols_with_insufficient_history(monkeypatch):
    monkeypatch.setattr(screener, "fetch_bars", lambda s, c, timeframe="1d": _bars(1.0, count=10))
    assert screener.scan("bist", Config(groq_api_key=""), "1d") == []
