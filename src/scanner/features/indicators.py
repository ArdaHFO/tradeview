"""Classic indicators: EMA, RSI (Wilder), ATR (Wilder), opening range.

Pure functions over lists of floats/Bars -- no pandas dependency, easy to test.
All functions return a list aligned with the input; warm-up positions are None.
"""
from __future__ import annotations

from ..models import Bar


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


def true_range(bar: Bar, prev_close: float | None) -> float:
    if prev_close is None:
        return bar.high - bar.low
    return max(bar.high - bar.low,
               abs(bar.high - prev_close),
               abs(bar.low - prev_close))


def atr(bars: list[Bar], period: int = 14) -> list[float | None]:
    """Wilder ATR over a list of bars."""
    out: list[float | None] = [None] * len(bars)
    if len(bars) < period + 1:
        return out
    trs = [true_range(bars[0], None)]
    for i in range(1, len(bars)):
        trs.append(true_range(bars[i], bars[i - 1].close))
    # Seed: simple average of first `period` TRs (skipping index 0's degenerate TR
    # would shift alignment; Wilder's original uses the first N TRs as-is).
    seed = sum(trs[1:period + 1]) / period
    out[period] = seed
    prev = seed
    for i in range(period + 1, len(bars)):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def opening_range(bars_1m: list[Bar], minutes: int = 15) -> tuple[float, float] | None:
    """High/low of the first `minutes` one-minute RTH bars.

    Caller must pass RTH bars only (session filtering is the caller's job).
    Returns None until enough bars exist.
    """
    if len(bars_1m) < minutes:
        return None
    window = bars_1m[:minutes]
    return max(b.high for b in window), min(b.low for b in window)


def rvol(day_cum_volume: float, avg_daily_volume: float,
         elapsed_fraction: float) -> float:
    """Relative volume, prorated by session elapsed fraction.

    rvol = today's cumulative volume / (avg full-day volume * elapsed fraction).
    Before the open (elapsed 0) we compare against 10% of a normal day, so a
    big pre-market print still registers as high RVOL instead of dividing by ~0.
    """
    if avg_daily_volume <= 0:
        return 0.0
    denom = avg_daily_volume * max(elapsed_fraction, 0.10)
    return day_cum_volume / denom


def bearish_divergence(prices: list[float], oscillator: list[float],
                       lookback: int = 20) -> bool:
    """Price makes a new high over `lookback` but the oscillator does not."""
    return _divergence(prices, oscillator, lookback, bearish=True)


def bullish_divergence(prices: list[float], oscillator: list[float],
                       lookback: int = 20) -> bool:
    """Price makes a new low over `lookback` but the oscillator does not."""
    return _divergence(prices, oscillator, lookback, bearish=False)


def _divergence(prices: list[float], oscillator: list[float],
                lookback: int, bearish: bool) -> bool:
    if len(prices) < lookback or len(oscillator) < lookback:
        return False
    p = prices[-lookback:]
    o = [v for v in oscillator[-lookback:] if v is not None]
    if len(o) < lookback:
        return False
    if bearish:
        # price at/near its window high, oscillator clearly below its window high
        return p[-1] >= max(p) and o[-1] < max(o[:-1])
    return p[-1] <= min(p) and o[-1] > min(o[:-1])
