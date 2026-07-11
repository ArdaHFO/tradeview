"""Rule-based technical confluence score in [-1, 1].

Weights: trend 40%, momentum 30%, volume confirmation 20%, Bollinger position 10%.
"""
from __future__ import annotations

from ..models import Bar, TechnicalVerdict
from .indicators import bollinger_bands, ema, macd, rsi, sma, volume_trend


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_technical(symbol: str, bars: list[Bar]) -> TechnicalVerdict:
    if len(bars) < 50:
        return TechnicalVerdict(symbol=symbol, score=0.0, reasons=["insufficient price history"])

    closes = [b.close for b in bars]
    volumes = [b.volume for b in bars]
    last = -1
    reasons: list[str] = []

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200) if len(closes) >= 200 else [None] * len(closes)
    ema20 = ema(closes, 20)
    rsi14 = rsi(closes, 14)
    _, _, hist = macd(closes)
    upper, mid, lower = bollinger_bands(closes)
    vtrend = volume_trend(volumes)

    trend_parts: list[float] = []
    if sma50[last] is not None and sma200[last] is not None:
        sign = 1.0 if sma50[last] > sma200[last] else -1.0
        trend_parts.append(sign)
        reasons.append(f"SMA50 {'>' if sign > 0 else '<'} SMA200")
    if ema20[last] is not None:
        sign = 1.0 if closes[last] > ema20[last] else -1.0
        trend_parts.append(sign)
        reasons.append(f"price {'above' if sign > 0 else 'below'} EMA20")
    trend_score = sum(trend_parts) / len(trend_parts) if trend_parts else 0.0

    momentum_parts: list[float] = []
    if rsi14[last] is not None:
        momentum_parts.append(_clamp((rsi14[last] - 50.0) / 50.0))
        reasons.append(f"RSI {rsi14[last]:.0f}")
    if hist[last] is not None:
        momentum_parts.append(1.0 if hist[last] > 0 else -1.0)
        reasons.append(f"MACD histogram {'positive' if hist[last] > 0 else 'negative'}")
    momentum_score = sum(momentum_parts) / len(momentum_parts) if momentum_parts else 0.0

    volume_score = 0.0
    if vtrend[last] is not None:
        direction = 1.0 if trend_score >= 0 else -1.0
        volume_score = direction * _clamp(vtrend[last] - 1.0)
        reasons.append(f"volume {vtrend[last]:.1f}x 20d avg")

    position_score = 0.0
    if upper[last] is not None and lower[last] is not None and mid[last] is not None:
        band_half_width = (upper[last] - lower[last]) / 2
        if band_half_width > 0:
            position_score = _clamp((closes[last] - mid[last]) / band_half_width)
            reasons.append(f"Bollinger position {position_score:+.2f}")

    final = _clamp(0.40 * trend_score + 0.30 * momentum_score
                    + 0.20 * volume_score + 0.10 * position_score)
    return TechnicalVerdict(symbol=symbol, score=final, reasons=reasons)
