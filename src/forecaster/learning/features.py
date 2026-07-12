"""Numeric feature vectors for the learned fusion model.

The same raw signals the rule-based scorer reads (trend / momentum / volume /
volatility), emitted as plain numbers a model can weight — plus two momentum
lookbacks. `news` is an optional feature: 0.0 when unknown (e.g. in historical
backtests where we don't have archived headlines), the live news score otherwise.

Indicators are computed once over the whole series (features_series), so a
backtest that reads a feature at every day stays linear, not quadratic.
"""
from __future__ import annotations

from ..models import Bar
from ..technical.indicators import (
    bollinger_bands, ema, macd, rsi, sma, supertrend, volume_trend,
)

# Order is the model's input order — never reorder without retraining.
FEATURE_NAMES: tuple[str, ...] = (
    "trend_sma",    # SMA50 vs SMA200 (+1/-1/0)
    "trend_ema",    # close vs EMA20
    "rsi",          # (RSI-50)/50, clipped to [-1,1]
    "macd_hist",    # sign of MACD histogram
    "bb_pos",       # position within Bollinger bands [-1,1]
    "vol_ratio",    # volume / 20d avg - 1, clipped
    "supertrend",   # +1/-1
    "ret_5",        # 5-bar return (scaled/clipped)
    "ret_20",       # 20-bar return (scaled/clipped)
    "news",         # news score [-1,1], 0 if unknown
)

MIN_BARS = 50


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def features_series(bars: list[Bar]) -> list[dict | None]:
    """Feature dict at every bar index (None during indicator warm-up), with
    the `news` feature left at 0.0 for callers to fill in."""
    n = len(bars)
    out: list[dict | None] = [None] * n
    if n < MIN_BARS:
        return out

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    vols = [b.volume for b in bars]

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200) if n >= 200 else [None] * n
    ema20 = ema(closes, 20)
    rsi14 = rsi(closes, 14)
    _, _, hist = macd(closes)
    up, mid, low = bollinger_bands(closes)
    vt = volume_trend(vols)
    _, _, st_dir, _ = supertrend(highs, lows, closes, period=10, multiplier=3.0)

    for i in range(n):
        if ema20[i] is None or rsi14[i] is None:
            continue  # core indicators not warmed up yet

        if sma50[i] is not None and sma200[i] is not None:
            trend_sma = 1.0 if sma50[i] > sma200[i] else -1.0
        else:
            trend_sma = 0.0
        trend_ema = 1.0 if closes[i] > ema20[i] else -1.0
        rsi_f = _clip((rsi14[i] - 50.0) / 50.0)
        macd_f = 0.0 if hist[i] is None else (1.0 if hist[i] > 0 else -1.0)
        if up[i] is not None and low[i] is not None and mid[i] is not None and (up[i] - low[i]) > 0:
            bb = _clip((closes[i] - mid[i]) / ((up[i] - low[i]) / 2.0))
        else:
            bb = 0.0
        vol = 0.0 if vt[i] is None else _clip(vt[i] - 1.0)
        st = 0.0 if st_dir[i] is None else float(st_dir[i])
        ret5 = _clip((closes[i] / closes[i - 5] - 1.0) * 5.0) if i >= 5 and closes[i - 5] else 0.0
        ret20 = _clip((closes[i] / closes[i - 20] - 1.0) * 2.0) if i >= 20 and closes[i - 20] else 0.0

        out[i] = {
            "trend_sma": trend_sma, "trend_ema": trend_ema, "rsi": rsi_f,
            "macd_hist": macd_f, "bb_pos": bb, "vol_ratio": vol,
            "supertrend": st, "ret_5": ret5, "ret_20": ret20, "news": 0.0,
        }
    return out


def features_from_bars(bars: list[Bar], news_score: float = 0.0) -> dict | None:
    """The single latest feature vector (for a live prediction), with `news`
    filled in from the current news score."""
    series = features_series(bars)
    latest = series[-1] if series else None
    if latest is None:
        return None
    latest = dict(latest)
    latest["news"] = _clip(news_score or 0.0)
    return latest


def to_vector(features: dict) -> list[float]:
    """Feature dict -> ordered numeric vector matching FEATURE_NAMES."""
    return [float(features[name]) for name in FEATURE_NAMES]
