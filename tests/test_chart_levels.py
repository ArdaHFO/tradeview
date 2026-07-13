from datetime import date, datetime, timedelta, timezone

from forecaster.models import Bar
from forecaster.webapp import _chart_summary, _fib_levels, _pivot_levels, _vwap_series


def test_pivot_levels_classic_formula():
    p = _pivot_levels(110.0, 90.0, 100.0)
    assert p == {"p": 100.0, "r1": 110.0, "s1": 90.0, "r2": 120.0, "s2": 80.0}


def test_fib_levels_between_high_and_low():
    f = _fib_levels(120.0, 80.0)
    assert f["0"] == 120.0 and f["100"] == 80.0
    assert f["50"] == 100.0            # midpoint
    assert f["38.2"] == 104.72         # 120 - 40*0.382


def test_chart_summary_includes_levels_and_position():
    closes = [10, 11, 12, 13, 14, 15, 16, 15, 14, 16, 18, 20]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    sm = _chart_summary(closes, highs, lows, [None] * len(closes))
    assert sm["period_high"] == 20 and sm["period_low"] == 10
    assert sm["position_pct"] == 100.0        # last close is the period high
    assert sm["pivot"] is not None and "p" in sm["pivot"]
    assert sm["fib"] is not None and sm["fib"]["0"] == 20


def test_chart_summary_empty_is_empty():
    assert _chart_summary([], [], [], []) == {}


def _bar(day: date, hour: int, close: float, volume: float = 100.0) -> Bar:
    ts = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    return Bar(ts=ts, open=close, high=close + 1, low=close - 1, close=close, volume=volume)


def test_vwap_is_between_low_and_high_of_the_session():
    d = date(2026, 1, 5)
    bars = [_bar(d, 9, 100.0, 10.0), _bar(d, 10, 102.0, 30.0), _bar(d, 11, 101.0, 20.0)]
    vwap = _vwap_series(bars)
    assert len(vwap) == 3
    assert all(v is not None for v in vwap)
    assert min(b.low for b in bars) <= vwap[-1] <= max(b.high for b in bars)


def test_vwap_resets_at_each_new_calendar_day():
    d1, d2 = date(2026, 1, 5), date(2026, 1, 6)
    bars = [_bar(d1, 9, 100.0, 10.0), _bar(d1, 15, 110.0, 10.0), _bar(d2, 9, 200.0, 10.0)]
    vwap = _vwap_series(bars)
    # Day 2's first bar starts a fresh accumulation — VWAP must equal that
    # bar's own typical price, not be dragged down by day 1's much lower prices.
    assert vwap[2] == round((201.0 + 199.0 + 200.0) / 3.0, 4)


def test_vwap_null_when_no_volume():
    bars = [_bar(date(2026, 1, 5), 9, 100.0, volume=0.0)]
    assert _vwap_series(bars) == [None]
