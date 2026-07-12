from forecaster.webapp import _chart_summary, _fib_levels, _pivot_levels


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
