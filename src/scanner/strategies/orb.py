"""Template C -- Opening Range Breakout: break of the first-15-min range,
validated by volume spike + CVD direction (order flow filters fake breakouts).

Two research-backed filters added after backtest showed the plain breakout
losing (PF 0.46 across two earlier fix attempts):
  1. VWAP directional bias -- a range that formed entirely above VWAP favors
     long breakouts, entirely below favors shorts (mixed range = no trade).
  2. Two-bar sustain -- the prior bar must also have closed beyond the range,
     so a single noise-tick poke through the level doesn't count as a breakout
     (a cheap proxy for "wait for a retest", the strongest false-breakout
     filter found in ORB literature).

Stop: opposite side of the opening range (capped at 1.5 ATR from entry).
"""
from __future__ import annotations

from datetime import datetime

from ..models import FamilyScores, SetupCandidate, SetupType, Side
from ..session import is_rth, minutes_since_open
from .base import Strategy, SymbolContext, clamp01

VOLUME_SPIKE_MULT = 3.0  # raised from 2.0 -- backtest showed weak spikes -> false breakouts
MIN_TAKER_IMBALANCE = 0.15  # sign-only agreement was too weak; require real magnitude
BREAKOUT_MARGIN_ATR = 0.1   # require a clear break past the range, not a tick over it


class OpeningRangeBreakout(Strategy):
    name = "orb"

    def evaluate(self, ctx: SymbolContext, now: datetime) -> SetupCandidate | None:
        or_minutes = self.cfg.opening_range_minutes
        if not is_rth(now) or minutes_since_open(now) <= or_minutes:
            return None
        orange = ctx.opening_range()
        if orange is None or not ctx.bars_1m:
            return None
        or_high, or_low = orange
        bar = ctx.bars_1m[-1]
        price = bar.close
        atr_1m = ctx.atr_now()
        avg_vol = ctx.avg_1m_volume()
        vwap = ctx.vwap.value
        if atr_1m is None or atr_1m <= 0 or avg_vol is None or avg_vol <= 0:
            return None

        vol_spike = bar.volume / avg_vol
        imb = ctx.flow.taker_imbalance()
        slope = ctx.flow.cvd_slope(minutes=3) or 0.0
        prev_bar = ctx.bars_1m[-2] if len(ctx.bars_1m) >= 2 else None

        if price > or_high + BREAKOUT_MARGIN_ATR * atr_1m:
            side = Side.LONG
            if vwap is not None and or_low <= vwap:
                return None                       # range not cleanly above VWAP: no long bias
            if prev_bar is None or prev_bar.close <= or_high:
                return None                       # need 2 consecutive closes past the level
            # order flow must clearly agree with the breakout direction
            if slope <= 0 or imb < MIN_TAKER_IMBALANCE:
                return None
            entry = price
            stop = max(or_low, entry - 1.5 * atr_1m)
            depth = (price - or_high) / atr_1m
            reasons = [f"broke OR high {or_high:.2f}"]
        elif price < or_low - BREAKOUT_MARGIN_ATR * atr_1m:
            side = Side.SHORT
            if vwap is not None and or_high >= vwap:
                return None                       # range not cleanly below VWAP: no short bias
            if prev_bar is None or prev_bar.close >= or_low:
                return None
            if slope >= 0 or imb > -MIN_TAKER_IMBALANCE:
                return None
            entry = price
            stop = min(or_high, entry + 1.5 * atr_1m)
            depth = (or_low - price) / atr_1m
            reasons = [f"broke OR low {or_low:.2f}"]
        else:
            return None

        if vol_spike < VOLUME_SPIKE_MULT:
            return None
        if abs(entry - stop) < 1e-9:
            return None
        reasons += [f"vol spike {vol_spike:.1f}x", f"taker imb {imb:+.2f}",
                    f"CVD slope {slope:+.0f}"]

        position_score = clamp01(0.5 + depth)                      # follow-through depth
        orderflow = clamp01(0.5 * clamp01(abs(imb) / 0.3)
                            + 0.5 * clamp01(abs(slope) / (3.0 * max(avg_vol, 1.0))))
        structure = clamp01(vol_spike / (2.0 * VOLUME_SPIKE_MULT) + 0.25)
        or_width_atr = (or_high - or_low) / atr_1m
        volatility = clamp01(or_width_atr / 4.0)                   # decent-width range
        scores = FamilyScores(position=position_score, orderflow=orderflow,
                              structure=structure, volatility=volatility)
        return SetupCandidate(symbol=ctx.symbol, setup=SetupType.ORB, side=side,
                              entry=entry, stop=stop, scores=scores, reasons=reasons)
