from datetime import datetime, timedelta

from scanner.features.indicators import (atr, bearish_divergence, ema,
                                         opening_range, rsi, rvol)
from scanner.models import Bar


def make_bars(closes, spread=1.0):
    t0 = datetime(2026, 6, 11, 13, 30)
    return [Bar(ts=t0 + timedelta(minutes=i), open=c, high=c + spread,
                low=c - spread, close=c, volume=1000.0)
            for i, c in enumerate(closes)]


def test_ema_seed_is_sma():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = ema(values, 3)
    assert out[0] is None and out[1] is None
    assert out[2] == 2.0                      # SMA(1,2,3)
    # next: (4 - 2) * 0.5 + 2 = 3.0
    assert abs(out[3] - 3.0) < 1e-9


def test_ema_constant_series():
    out = ema([5.0] * 10, 4)
    assert all(abs(v - 5.0) < 1e-9 for v in out if v is not None)


def test_rsi_all_gains_is_100():
    closes = [float(i) for i in range(1, 20)]
    out = rsi(closes, 14)
    assert out[14] == 100.0


def test_rsi_known_value():
    # Standard Wilder example dataset
    closes = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
              45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28]
    out = rsi(closes, 14)
    assert out[14] is not None
    assert 69.0 < out[14] < 71.0   # canonical ~70.46


def test_atr_constant_range():
    bars = make_bars([10.0] * 20, spread=0.5)   # TR = 1.0 every bar
    out = atr(bars, 14)
    assert out[14] is not None
    assert abs(out[14] - 1.0) < 1e-9
    assert abs(out[-1] - 1.0) < 1e-9


def test_opening_range():
    bars = make_bars([10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 14, 15, 14, 13, 12])
    orange = opening_range(bars, 15)
    assert orange == (16.0, 8.0)   # max high = 15+1, min low = 9-1
    assert opening_range(bars[:10], 15) is None


def test_rvol_prorated():
    # avg full day 1M, 50% of session elapsed, 1M already traded -> rvol 2.0
    assert abs(rvol(1_000_000, 1_000_000, 0.5) - 2.0) < 1e-9
    # before open: floor at 10% of normal day
    assert abs(rvol(200_000, 1_000_000, 0.0) - 2.0) < 1e-9
    assert rvol(100, 0.0, 0.5) == 0.0


def test_bearish_divergence():
    # price grinds to a new high while oscillator fades
    prices = [10 + 0.1 * i for i in range(20)]
    osc = [80.0 - i for i in range(20)]
    assert bearish_divergence(prices, osc, lookback=20)
    # no divergence when oscillator confirms
    osc_up = [50.0 + i for i in range(20)]
    assert not bearish_divergence(prices, osc_up, lookback=20)
