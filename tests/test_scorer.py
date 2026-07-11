from datetime import datetime, timedelta, timezone

from forecaster.models import Bar
from forecaster.technical.scorer import score_technical


def _make_bars(closes: list[float], volume: float = 1_000_000.0) -> list[Bar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(ts=start + timedelta(days=i), open=c, high=c * 1.01, low=c * 0.99,
            close=c, volume=volume)
        for i, c in enumerate(closes)
    ]


def test_uptrend_scores_positive():
    closes = [100 + i * 0.5 for i in range(220)]  # steady uptrend, 220 days
    bars = _make_bars(closes)
    verdict = score_technical("TEST", bars)
    assert verdict.score > 0


def test_downtrend_scores_negative():
    closes = [200 - i * 0.5 for i in range(220)]  # steady downtrend
    bars = _make_bars(closes)
    verdict = score_technical("TEST", bars)
    assert verdict.score < 0


def test_insufficient_history_scores_zero():
    closes = [100.0] * 10
    bars = _make_bars(closes)
    verdict = score_technical("TEST", bars)
    assert verdict.score == 0.0
    assert "insufficient" in verdict.reasons[0]
