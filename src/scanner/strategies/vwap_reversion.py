"""Template B -- VWAP Reversion: fade an overextended move back to VWAP.

Trigger (SHORT side; mirror for LONG):
  price >= VWAP + 2 ATR AND CVD bearish divergence (price high, CVD lower high)
  AND tape slowing down AND a rejection wick on the latest bar -> fade toward VWAP.
Stop: beyond the extreme by 0.5 ATR. Target: VWAP (overrides the default 2R).

Tape-slowing and rejection-wick were previously soft (scoring-only) signals;
research on mean-reversion fades treats "decreasing volume / exhaustion candle"
and "rejection wick" as the actual entry trigger, not a nice-to-have, so both
are now hard gates alongside the CVD divergence check.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..features.orderflow import cvd_bearish_divergence, cvd_bullish_divergence
from ..models import FamilyScores, SetupCandidate, SetupType, Side
from ..session import is_rth, minutes_since_open
from .base import Strategy, SymbolContext, clamp01

MIN_REJECTION_WICK_FRAC = 0.25  # opposite-side wick must be >= 25% of the bar's range


class VwapReversion(Strategy):
    name = "vwap_reversion"

    def evaluate(self, ctx: SymbolContext, now: datetime) -> SetupCandidate | None:
        # Mid-day setup: skip the chaotic first 30 minutes.
        if not is_rth(now) or minutes_since_open(now) < 30:
            return None
        price = ctx.last_price
        vwap = ctx.vwap.value
        atr_1m = ctx.atr_now()
        if price is None or vwap is None or atr_1m is None or atr_1m <= 0:
            return None

        dist_atr = (price - vwap) / atr_1m
        min_dist = self.cfg.vwap_reversion_atr_dist
        if abs(dist_atr) < min_dist:
            return None

        closes = ctx.closes
        cvd_series = ctx.flow.cvd_minute_series()
        tape_now = ctx.flow.tape_speed(now)
        tape_before = ctx.flow.tape_speed(now - timedelta(minutes=2))
        tape_slowing = tape_before > 0 and tape_now < tape_before
        if not tape_slowing:
            return None                           # exhaustion (slowing tape) is now required, not scored

        last_bar = ctx.bars_1m[-1]
        bar_range = last_bar.high - last_bar.low
        if bar_range <= 0:
            return None

        if dist_atr > 0:
            # extended above VWAP -> look for a SHORT fade
            if not cvd_bearish_divergence(closes, cvd_series):
                return None
            rejection = (last_bar.high - last_bar.close) / bar_range
            if rejection < MIN_REJECTION_WICK_FRAC:
                return None                       # no upper-wick rejection: momentum hasn't stalled
            side = Side.SHORT
            extreme = max(b.high for b in ctx.bars_1m[-10:])
            stop = extreme + 0.5 * atr_1m
            reasons = [f"{dist_atr:.1f} ATR above VWAP", "CVD bearish divergence",
                       f"rejection wick {rejection:.0%}"]
        else:
            if not cvd_bullish_divergence(closes, cvd_series):
                return None
            rejection = (last_bar.close - last_bar.low) / bar_range
            if rejection < MIN_REJECTION_WICK_FRAC:
                return None
            side = Side.LONG
            extreme = min(b.low for b in ctx.bars_1m[-10:])
            stop = extreme - 0.5 * atr_1m
            reasons = [f"{abs(dist_atr):.1f} ATR below VWAP", "CVD bullish divergence",
                       f"rejection wick {rejection:.0%}"]

        entry = price
        if abs(entry - stop) < 1e-9:
            return None
        reasons.append("tape slowing")

        # Family scores
        position = clamp01((abs(dist_atr) - min_dist) / min_dist + 0.5)
        orderflow = clamp01(0.85 + 0.15 * clamp01(rejection))  # divergence + tape + wick already required
        structure = clamp01(0.5 + 0.5 * clamp01(rejection))
        volatility = clamp01(abs(dist_atr) / (2.0 * min_dist))
        scores = FamilyScores(position=position, orderflow=orderflow,
                              structure=structure, volatility=volatility)
        cand = SetupCandidate(symbol=ctx.symbol, setup=SetupType.VWAP_REVERSION,
                              side=side, entry=entry, stop=stop,
                              scores=scores, reasons=reasons)
        return cand
