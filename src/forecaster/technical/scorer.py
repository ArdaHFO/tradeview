"""Rule-based technical confluence score in [-1, 1].

Weights: trend 40% (SMA50/200 20% + EMA20 20%), momentum 25% (RSI 12.5% +
MACD 12.5%), volume confirmation 15%, Bollinger position 10%, Supertrend 10%.
"""
from __future__ import annotations

from ..models import Bar, Direction, IndicatorDetail, TechnicalVerdict
from .indicators import bollinger_bands, ema, macd, rsi, sma, supertrend, volume_trend


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _dir(sign: float) -> Direction:
    if sign > 0:
        return Direction.UP
    if sign < 0:
        return Direction.DOWN
    return Direction.NEUTRAL


def score_technical(symbol: str, bars: list[Bar]) -> TechnicalVerdict:
    if len(bars) < 50:
        return TechnicalVerdict(symbol=symbol, score=0.0, reasons=["insufficient price history"])

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]
    last = -1
    reasons: list[str] = []
    indicators: list[IndicatorDetail] = []

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200) if len(closes) >= 200 else [None] * len(closes)
    ema20 = ema(closes, 20)
    rsi14 = rsi(closes, 14)
    _, _, hist = macd(closes)
    upper, mid, lower = bollinger_bands(closes)
    vtrend = volume_trend(volumes)
    _, _, st_dir, st_line = supertrend(highs, lows, closes, period=10, multiplier=3.0)

    trend_parts: list[float] = []
    if sma50[last] is not None and sma200[last] is not None:
        sign = 1.0 if sma50[last] > sma200[last] else -1.0
        trend_parts.append(sign)
        reasons.append(f"SMA50 {'>' if sign > 0 else '<'} SMA200")
        indicators.append(IndicatorDetail(
            name="SMA50 / SMA200", value=f"{sma50[last]:.2f} / {sma200[last]:.2f}",
            direction=_dir(sign), weight_pct=20.0,
            explanation="50 günlük ortalama 200 günlük ortalamanın üzerindeyse (Golden "
                        "Cross) uzun vadeli yükseliş, altındaysa (Death Cross) uzun vadeli "
                        "düşüş eğilimine işaret eder.",
        ))
    if ema20[last] is not None:
        sign = 1.0 if closes[last] > ema20[last] else -1.0
        trend_parts.append(sign)
        reasons.append(f"price {'above' if sign > 0 else 'below'} EMA20")
        indicators.append(IndicatorDetail(
            name="Fiyat / EMA20", value=f"{closes[last]:.2f} / {ema20[last]:.2f}",
            direction=_dir(sign), weight_pct=20.0,
            explanation="Fiyatın 20 günlük üstel hareketli ortalamanın üzerinde olması "
                        "kısa vadeli yükseliş, altında olması kısa vadeli düşüş eğilimini gösterir.",
        ))
    trend_score = sum(trend_parts) / len(trend_parts) if trend_parts else 0.0

    # Supertrend is scored as its own 10% term below, not folded into
    # trend_score, so it isn't counted twice.
    supertrend_score = 0.0
    if st_dir[last] is not None and st_line[last] is not None:
        supertrend_score = float(st_dir[last])
        reasons.append(f"Supertrend {'bullish' if supertrend_score > 0 else 'bearish'}")
        indicators.append(IndicatorDetail(
            name="Supertrend", value=("Yükseliş (Bullish)" if supertrend_score > 0 else "Düşüş (Bearish)"),
            direction=_dir(supertrend_score), weight_pct=10.0,
            explanation="ATR (ortalama gerçek aralık) tabanlı trend takip göstergesi; "
                        "fiyatın Supertrend çizgisinin üzerinde/altında olmasına göre yön belirler.",
        ))

    momentum_parts: list[float] = []
    if rsi14[last] is not None:
        rsi_signed = _clamp((rsi14[last] - 50.0) / 50.0)
        momentum_parts.append(rsi_signed)
        reasons.append(f"RSI {rsi14[last]:.0f}")
        indicators.append(IndicatorDetail(
            name="RSI (14)", value=f"{rsi14[last]:.1f}",
            direction=_dir(rsi14[last] - 50.0), weight_pct=12.5,
            explanation="Göreceli Güç Endeksi. 70 üzeri aşırı alım, 30 altı aşırı satım "
                        "bölgesidir; 50 üzeri momentumun yukarı, 50 altı aşağı yönlü olduğunu gösterir.",
        ))
    if hist[last] is not None:
        momentum_parts.append(1.0 if hist[last] > 0 else -1.0)
        reasons.append(f"MACD histogram {'positive' if hist[last] > 0 else 'negative'}")
        indicators.append(IndicatorDetail(
            name="MACD Histogram", value=f"{hist[last]:+.3f}",
            direction=_dir(hist[last]), weight_pct=12.5,
            explanation="MACD çizgisi ile sinyal çizgisi arasındaki fark. Pozitifse kısa "
                        "vadeli momentum güçleniyor, negatifse zayıflıyor demektir.",
        ))
    momentum_score = sum(momentum_parts) / len(momentum_parts) if momentum_parts else 0.0

    volume_score = 0.0
    if vtrend[last] is not None:
        direction = 1.0 if trend_score >= 0 else -1.0
        volume_score = direction * _clamp(vtrend[last] - 1.0)
        reasons.append(f"volume {vtrend[last]:.1f}x 20d avg")
        indicators.append(IndicatorDetail(
            name="Hacim Oranı", value=f"{vtrend[last]:.2f}x (20g ort.)",
            direction=_dir(volume_score), weight_pct=15.0,
            explanation="Son işlem hacminin 20 günlük ortalamaya oranı. Ortalamanın "
                        "üzerinde hacim, fiyat hareketinin gücünü teyit eder.",
        ))

    position_score = 0.0
    if upper[last] is not None and lower[last] is not None and mid[last] is not None:
        band_half_width = (upper[last] - lower[last]) / 2
        if band_half_width > 0:
            position_score = _clamp((closes[last] - mid[last]) / band_half_width)
            reasons.append(f"Bollinger position {position_score:+.2f}")
            indicators.append(IndicatorDetail(
                name="Bollinger Bantları", value=f"{position_score:+.2f} (bant içi konum)",
                direction=_dir(position_score), weight_pct=10.0,
                explanation="Fiyatın bantlar içindeki konumu -1 (alt bant) ile +1 (üst bant) "
                            "arasında ölçülür. Üst banda yakınlık aşırı alım, alt banda "
                            "yakınlık aşırı satım baskısına işaret edebilir.",
            ))

    final = _clamp(0.40 * trend_score + 0.25 * momentum_score
                    + 0.15 * volume_score + 0.10 * position_score
                    + 0.10 * supertrend_score)
    return TechnicalVerdict(symbol=symbol, score=final, reasons=reasons, indicators=indicators)
