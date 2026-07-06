"""Template B -- VWAP Reversion: fade an overextended move back to VWAP.

Trigger (SHORT side; mirror for LONG):
  price >= VWAP + 2 ATR AND CVD bearish divergence (price high, CVD lower high)
  AND tape slowing down -> fade toward VWAP.
Stop: beyond the extreme by 0.5 ATR. Target: VWAP (overrides the default 2R).
"""
from __future__ import annotations

from datetime import datetime, timedelta

from ..features.orderflow import cvd_bearish_divergence, cvd_bullish_divergence
from ..models import FamilyScores, SetupCandidate, SetupType, Side
from ..session import is_rth, minutes_since_open
from .base import Strategy, SymbolContext, clamp01


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

        if dist_atr > 0:
            # extended above VWAP -> look for a SHORT fade
            if not cvd_bearish_divergence(closes, cvd_series):
                return None
            side = Side.SHORT
            extreme = max(b.high for b in ctx.bars_1m[-10:])
            stop = extreme + 0.5 * atr_1m
            reasons = [f"{dist_atr:.1f} ATR above VWAP", "CVD bearish divergence"]
        else:
            if not cvd_bullish_divergence(closes, cvd_series):
                return None
            side = Side.LONG
            extreme = min(b.low for b in ctx.bars_1m[-10:])
            stop = extreme - 0.5 * atr_1m
            reasons = [f"{abs(dist_atr):.1f} ATR below VWAP", "CVD bullish divergence"]

        entry = price
        if abs(entry - stop) < 1e-9:
            return None
        if tape_slowing:
            reasons.append("tape slowing")

        # Family scores
        position = clamp01((abs(dist_atr) - min_dist) / min_dist + 0.5)
        orderflow = clamp01(0.7 + (0.3 if tape_slowing else 0.0))  # divergence already required
        structure = clamp01(0.5 + (0.5 if tape_slowing else 0.0))
        volatility = clamp01(abs(dist_atr) / (2.0 * min_dist))
        scores = FamilyScores(position=position, orderflow=orderflow,
                              structure=structure, volatility=volatility)
        cand = SetupCandidate(symbol=ctx.symbol, setup=SetupType.VWAP_REVERSION,
                              side=side, entry=entry, stop=stop,
                              scores=scores, reasons=reasons)
        return cand
