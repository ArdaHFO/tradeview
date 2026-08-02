"""Swing strategies on daily bars, each a documented published edge.

These are deliberately not new inventions. The intraday setups in this repo were
hand-tuned until the backtest looked good and then failed statistical validation,
which is the classic signature of fitting noise. So each strategy here is a rule
set with published prior evidence, implemented as specified rather than tuned to
this sample, and then put through the same bootstrap/Monte Carlo test. If a
strategy with real prior support still fails here, that is informative about the
universe or the data; if a hand-tuned one passes, it usually just means we tried
enough variants.

  * `meanrev`  -- Connors RSI(2): in a long-term uptrend, buy severe short-term
                  oversold, exit on the bounce. Documented in Connors & Alvarez,
                  *Short Term Trading Strategies That Work*.
  * `breakout` -- Donchian channel breakout with an ATR stop, the core of the
                  Turtle system; the classic trend-following entry.
  * `trend`    -- pullback continuation: established uptrend, price dips to the
                  20-day mean and turns back up.

Lookahead discipline: every signal is computed from data up to and including
day D, and the backtester fills it at the OPEN of D+1. Nothing here may read a
future bar.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# --- indicators (vectorised, Wilder smoothing where the original spec uses it) ---

def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(100.0)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"],
                    (df["High"] - prev_close).abs(),
                    (df["Low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    out["sma5"] = close.rolling(5).mean()
    out["sma20"] = close.rolling(20).mean()
    out["sma50"] = close.rolling(50).mean()
    out["sma200"] = close.rolling(200).mean()
    out["rsi2"] = rsi(close, 2)
    out["atr14"] = atr(out, 14)
    # shift(1) so the channel excludes today: "highest high of the PRIOR 55 days"
    out["dc_high55"] = out["High"].rolling(55).max().shift(1)
    out["dc_low20"] = out["Low"].rolling(20).min().shift(1)
    out["dollar_vol20"] = (close * out["Volume"]).rolling(20).mean()
    return out


# --- signals --------------------------------------------------------------

class Strategy:
    """Vectorised signal generator.

    `entries()` returns a frame indexed by signal date with a `rank` column
    (higher wins when more signals fire than the portfolio has slots) and a
    `reason` column. Everything is computed column-wise: a per-row Python loop
    over 500 symbols x 10 years is minutes of pure overhead.
    """
    name = "base"
    stop_atr = 3.0              # initial stop distance in ATRs
    max_hold_days = 20          # time stop
    exit_rule = "sma5"          # how the position is managed out

    def entries(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def _frame(mask: pd.Series, rank: pd.Series, reason: pd.Series) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        return pd.DataFrame({"rank": rank[mask], "reason": reason[mask]})


class MeanReversion(Strategy):
    """Connors RSI(2): buy deep oversold inside a long-term uptrend."""
    name = "meanrev"
    stop_atr = 3.0
    max_hold_days = 10
    exit_rule = "sma5"          # exit when close reclaims the 5-day mean
    RSI_ENTRY = 5.0

    def entries(self, df):
        mask = (df["Close"] > df["sma200"]) & (df["rsi2"] < self.RSI_ENTRY)
        reason = "RSI2 " + df["rsi2"].round(1).astype(str) + ", 200MA üstü"
        return self._frame(mask, -df["rsi2"], reason)   # deeper oversold ranks first


class Breakout(Strategy):
    """Donchian 55-day breakout with an ATR stop (Turtle entry)."""
    name = "breakout"
    stop_atr = 2.0
    max_hold_days = 60
    exit_rule = "dc_low20"      # trail out on a 20-day low

    def entries(self, df):
        broke = df["Close"] > df["dc_high55"]
        # only the first bar of a breakout counts, not every day above the channel
        fresh = broke & ~(df["Close"].shift(1) > df["dc_high55"].shift(1))
        mask = fresh & (df["Close"] > df["sma200"])
        strength = (df["Close"] - df["dc_high55"]) / df["atr14"].clip(lower=1e-9)
        reason = "55g kırılım +" + strength.round(2).astype(str) + " ATR"
        return self._frame(mask, strength, reason)


class TrendPullback(Strategy):
    """Buy the first turn up from the 20-day mean inside an established uptrend."""
    name = "trend"
    stop_atr = 2.5
    max_hold_days = 30
    exit_rule = "sma20"

    def entries(self, df):
        uptrend = (df["sma50"] > df["sma200"]) & (df["Close"] > df["sma200"])
        dipped = df["Close"].shift(1) < df["sma20"].shift(1)
        turned = df["Close"] > df["sma20"]
        mask = uptrend & dipped & turned
        depth = (df["sma50"] - df["sma200"]) / df["sma200"].clip(lower=1e-9)
        reason = pd.Series("20MA'ya çekilip döndü, 50>200", index=df.index)
        return self._frame(mask, depth, reason)


REGISTRY: dict[str, type[Strategy]] = {
    MeanReversion.name: MeanReversion,
    Breakout.name: Breakout,
    TrendPullback.name: TrendPullback,
}


def build(name: str) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"bilinmeyen strateji '{name}' "
                       f"(seçenekler: {', '.join(sorted(REGISTRY))})")
    return REGISTRY[name]()
