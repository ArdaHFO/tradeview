"""Classic indicators: EMA, SMA, RSI (Wilder), MACD, Bollinger Bands, Supertrend, volume trend.

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


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 10) -> list[float | None]:
    """Average True Range using Wilder smoothing."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must have the same length")
    if period <= 0:
        raise ValueError("period must be positive")
    if len(closes) < period + 1:
        return [None] * len(closes)

    tr: list[float] = [highs[0] - lows[0]]
    for i in range(1, len(closes)):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    out: list[float | None] = [None] * len(closes)
    seed = sum(tr[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(closes)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def supertrend(highs: list[float], lows: list[float], closes: list[float], period: int = 10,
               multiplier: float = 3.0) -> tuple[list[float | None], list[float | None], list[int | None], list[float | None]]:
    """Returns (upper_band, lower_band, direction, supertrend_line). Direction is 1 for up, -1 for down."""
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, and closes must have the same length")
    atr_values = atr(highs, lows, closes, period)
    upper_band: list[float | None] = [None] * len(closes)
    lower_band: list[float | None] = [None] * len(closes)
    direction: list[int | None] = [None] * len(closes)
    line: list[float | None] = [None] * len(closes)

    for i in range(len(closes)):
        atr_value = atr_values[i]
        if atr_value is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2.0
        basic_upper = hl2 + multiplier * atr_value
        basic_lower = hl2 - multiplier * atr_value

        if i == 0:
            upper_band[i] = basic_upper
            lower_band[i] = basic_lower
            direction[i] = 1
            line[i] = basic_lower
            continue

        prev_upper = upper_band[i - 1] if upper_band[i - 1] is not None else basic_upper
        prev_lower = lower_band[i - 1] if lower_band[i - 1] is not None else basic_lower
        prev_close = closes[i - 1]

        upper_band[i] = basic_upper if basic_upper < prev_upper or prev_close > prev_upper else prev_upper
        lower_band[i] = basic_lower if basic_lower > prev_lower or prev_close < prev_lower else prev_lower

        prev_direction = direction[i - 1] or 1
        if closes[i] > prev_upper:
            direction[i] = 1
        elif closes[i] < prev_lower:
            direction[i] = -1
        else:
            direction[i] = prev_direction

        line[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    return upper_band, lower_band, direction, line
