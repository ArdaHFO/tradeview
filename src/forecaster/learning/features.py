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
    atr, bollinger_bands, ema, macd, rsi, sma, supertrend, volume_trend,
)

# Order is the model's input order — never reorder without retraining. The
# model standardises every column, so raw scales (returns, ATR%) are fine.
# (Loaded models map features BY NAME via predict_from_dict, so adding features
# here never breaks an older model.json.)
FEATURE_NAMES: tuple[str, ...] = (
    "trend_sma",    # SMA50 vs SMA200 (+1/-1/0)
    "trend_ema",    # close vs EMA20
    "rsi",          # (RSI-50)/50, clipped to [-1,1]
    "macd_hist",    # sign of MACD histogram
    "bb_pos",       # position within Bollinger bands [-1,1]
    "vol_ratio",    # volume / 20d avg - 1, clipped
    "supertrend",   # +1/-1
    "ret_5",        # 5-bar return
    "ret_20",       # 20-bar return (~1 month)
    "ret_60",       # 60-bar return (~3 months momentum)
    "ret_120",      # 120-bar return (~6 months momentum)
    "dist_high",    # distance below the 52-week high, in [-1, 0]
    "ma_gap",       # (close - SMA50) / SMA50, continuous trend strength
    "atr_pct",      # ATR / close — volatility regime
    "mkt_ret_20",   # market index 20-bar return — regime ("don't fight the tape")
    "mkt_trend",    # market index above/below its SMA50 (+1/-1, 0 unknown)
    "rel_ret_60",   # stock 60-bar return MINUS market's — cross-sectional momentum
    "news",         # news score [-1,1], 0 if unknown
)

MIN_BARS = 50
_HIGH_WINDOW = 252  # ~1 trading year for the 52-week high

# Which market index proxies "the tape" for a given exchange suffix.
_MARKET_INDEX_BY_SUFFIX: dict[str, str] = {
    "IS": "XU100.IS",   # BIST 100
    "DE": "^GDAXI", "F": "^GDAXI",
    "PA": "^FCHI",
    "L": "^FTSE",
    "AS": "^AEX",
    "MI": "FTSEMIB.MI",
    "MC": "^IBEX",
    "SW": "^SSMI",
    "T": "^N225",
    "HK": "^HSI",
    "KS": "^KS11",
    "SA": "^BVSP",
}
_DEFAULT_MARKET_INDEX = "^GSPC"  # S&P 500 for unsuffixed (US) tickers


def market_index_for(symbol: str) -> str:
    """The market index whose regime this symbol trades under."""
    if "." not in symbol:
        return _DEFAULT_MARKET_INDEX
    suffix = symbol.rsplit(".", 1)[-1].upper()
    return _MARKET_INDEX_BY_SUFFIX.get(suffix, _DEFAULT_MARKET_INDEX)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _market_context(bars: list[Bar], market_bars: list[Bar] | None):
    """Per-bar (mkt_ret_20, mkt_trend, market_ret_60_raw) aligned by date.

    Both series are chronological, so a single forward-walking pointer aligns
    each stock bar with the latest market bar at-or-before it (no look-ahead).
    Returns None when there's no usable market series — features stay 0.0.
    """
    if not market_bars or len(market_bars) < 60:
        return None
    mcloses = [b.close for b in market_bars]
    msma50 = sma(mcloses, 50)
    out = []
    j = 0
    for bar in bars:
        while j + 1 < len(market_bars) and market_bars[j + 1].ts <= bar.ts:
            j += 1
        if market_bars[j].ts > bar.ts:      # market history starts after this bar
            out.append((0.0, 0.0, None))
            continue
        ret20 = _clip(mcloses[j] / mcloses[j - 20] - 1.0) if j >= 20 and mcloses[j - 20] else 0.0
        trend = 0.0
        if msma50[j] is not None:
            trend = 1.0 if mcloses[j] > msma50[j] else -1.0
        ret60_raw = (mcloses[j] / mcloses[j - 60] - 1.0) if j >= 60 and mcloses[j - 60] else None
        out.append((ret20, trend, ret60_raw))
    return out


def features_series(bars: list[Bar], market_bars: list[Bar] | None = None) -> list[dict | None]:
    """Feature dict at every bar index (None during indicator warm-up), with
    the `news` feature left at 0.0 for callers to fill in. `market_bars` (the
    symbol's market index, same timeframe) powers the regime/relative-strength
    features; without it those stay 0.0 (neutral)."""
    n = len(bars)
    out: list[dict | None] = [None] * n
    if n < MIN_BARS:
        return out

    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    vols = [b.volume for b in bars]
    mkt = _market_context(bars, market_bars)

    sma50 = sma(closes, 50)
    sma200 = sma(closes, 200) if n >= 200 else [None] * n
    ema20 = ema(closes, 20)
    rsi14 = rsi(closes, 14)
    _, _, hist = macd(closes)
    up, mid, low = bollinger_bands(closes)
    vt = volume_trend(vols)
    _, _, st_dir, _ = supertrend(highs, lows, closes, period=10, multiplier=3.0)
    atr14 = atr(highs, lows, closes, 14)

    for i in range(n):
        recent_high = max(closes[max(0, i - _HIGH_WINDOW + 1):i + 1])  # trailing 52-week high

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
        ret5 = _clip(closes[i] / closes[i - 5] - 1.0) if i >= 5 and closes[i - 5] else 0.0
        ret20 = _clip(closes[i] / closes[i - 20] - 1.0) if i >= 20 and closes[i - 20] else 0.0
        ret60_raw = (closes[i] / closes[i - 60] - 1.0) if i >= 60 and closes[i - 60] else None
        ret60 = _clip(ret60_raw) if ret60_raw is not None else 0.0
        ret120 = _clip(closes[i] / closes[i - 120] - 1.0) if i >= 120 and closes[i - 120] else 0.0
        dist_high = _clip((closes[i] - recent_high) / recent_high) if recent_high else 0.0
        ma_gap = _clip((closes[i] - sma50[i]) / sma50[i]) if sma50[i] else 0.0
        atr_pct = _clip(atr14[i] / closes[i], 0.0, 1.0) if atr14[i] is not None and closes[i] else 0.0

        mkt_ret_20, mkt_trend, rel_ret_60 = 0.0, 0.0, 0.0
        if mkt is not None:
            m20, mtrend, m60_raw = mkt[i]
            mkt_ret_20, mkt_trend = m20, mtrend
            if m60_raw is not None and ret60_raw is not None:
                rel_ret_60 = _clip(ret60_raw - m60_raw)

        out[i] = {
            "trend_sma": trend_sma, "trend_ema": trend_ema, "rsi": rsi_f,
            "macd_hist": macd_f, "bb_pos": bb, "vol_ratio": vol, "supertrend": st,
            "ret_5": ret5, "ret_20": ret20, "ret_60": ret60, "ret_120": ret120,
            "dist_high": dist_high, "ma_gap": ma_gap, "atr_pct": atr_pct,
            "mkt_ret_20": mkt_ret_20, "mkt_trend": mkt_trend, "rel_ret_60": rel_ret_60,
            "news": 0.0,
        }
    return out


def features_from_bars(bars: list[Bar], news_score: float = 0.0,
                       market_bars: list[Bar] | None = None) -> dict | None:
    """The single latest feature vector (for a live prediction), with `news`
    filled in from the current news score."""
    series = features_series(bars, market_bars)
    latest = series[-1] if series else None
    if latest is None:
        return None
    latest = dict(latest)
    latest["news"] = _clip(news_score or 0.0)
    return latest


def to_vector(features: dict) -> list[float]:
    """Feature dict -> ordered numeric vector matching FEATURE_NAMES."""
    return [float(features[name]) for name in FEATURE_NAMES]
