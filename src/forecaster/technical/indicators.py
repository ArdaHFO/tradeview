"""Classic indicators: EMA, SMA, RSI (Wilder), MACD, Bollinger Bands, volume trend.

Pure functions over lists of floats -- no pandas dependency, easy to test.
All functions return a list aligned with the input; warm-up positions are None.
"""
from __future__ import annotations


def sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1:i + 1]) / period
    return out


def ema(values: list[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the SMA of the first `period` values."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    k = 2.0 / (period + 1)
    prev = seed
    for i in range(period, len(values)):
        prev = (values[i] - prev) * k + prev
        out[i] = prev
    return out


def rsi(closes: list[float], period: int = 14) -> list[float | None]:
    """Wilder RSI."""
    out: list[float | None] = [None] * len(closes)
    if len(closes) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def macd(closes: list[float], fast: int = 12, slow: int = 26,
         signal: int = 9) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line: list[float | None] = [
        (f - s) if f is not None and s is not None else None
        for f, s in zip(ema_fast, ema_slow)
    ]
    macd_values = [v for v in macd_line if v is not None]
    signal_seed_offset = len(macd_line) - len(macd_values)
    signal_ema = ema(macd_values, signal) if len(macd_values) >= signal else [None] * len(macd_values)
    signal_line: list[float | None] = [None] * signal_seed_offset + signal_ema
    histogram: list[float | None] = [
        (m - s) if m is not None and s is not None else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, histogram


def bollinger_bands(closes: list[float], period: int = 20,
                     mult: float = 2.0) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Returns (upper, mid, lower)."""
    mid = sma(closes, period)
    upper: list[float | None] = [None] * len(closes)
    lower: list[float | None] = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        m = mid[i]
        assert m is not None
        variance = sum((x - m) ** 2 for x in window) / period
        std = variance ** 0.5
        upper[i] = m + mult * std
        lower[i] = m - mult * std
    return upper, mid, lower


def volume_trend(volumes: list[float], period: int = 20) -> list[float | None]:
    """Ratio of each day's volume to the trailing `period`-day average volume."""
    avg = sma(volumes, period)
    return [
        (v / a) if a is not None and a > 0 else None
        for v, a in zip(volumes, avg)
    ]
