import pytest

from forecaster.technical.indicators import (
    bollinger_bands, ema, macd, rsi, sma, volume_trend,
)


def test_sma_basic():
    values = [1, 2, 3, 4, 5]
    out = sma(values, 3)
    assert out[:2] == [None, None]
    assert out[2] == 2.0
    assert out[3] == 3.0
    assert out[4] == 4.0


def test_ema_seeds_with_sma_then_tracks():
    values = [10, 10, 10, 10, 20]
    out = ema(values, 4)
    assert out[3] == 10.0
    assert out[4] is not None and out[4] > 10.0


def test_rsi_all_gains_is_100():
    closes = [float(i) for i in range(1, 20)]  # strictly increasing
    out = rsi(closes, period=14)
    assert out[14] == 100.0


def test_rsi_all_losses_is_0():
    closes = [float(i) for i in range(20, 1, -1)]  # strictly decreasing
    out = rsi(closes, period=14)
    assert out[14] == 0.0


def test_macd_returns_aligned_lists():
    closes = [float(100 + i) for i in range(40)]
    macd_line, signal_line, hist = macd(closes, fast=12, slow=26, signal=9)
    assert len(macd_line) == len(signal_line) == len(hist) == len(closes)
    assert macd_line[-1] is not None
    assert hist[-1] is not None


def test_bollinger_bands_upper_gt_lower():
    closes = [100 + (i % 5) for i in range(30)]
    upper, mid, lower = bollinger_bands(closes, period=20)
    assert upper[-1] > mid[-1] > lower[-1]


def test_volume_trend_ratio():
    volumes = [100.0] * 20 + [300.0]
    out = volume_trend(volumes, period=20)
    # window at the last index is volumes[1:21]: 19x100 + 1x300, avg=110
    assert out[-1] == pytest.approx(300.0 / 110.0)
