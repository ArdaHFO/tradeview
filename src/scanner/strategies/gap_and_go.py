"""Template A -- Gap-and-Go: opening momentum continuation on a gapped, in-play name.

LONG trigger (mirror for SHORT on a gap-down):
  gap >= +4% (screener) AND first 60 min AND price above VWAP
  AND CVD positive & rising AND price breaks the first-5-minute high.
Stop: max(VWAP, 5-min low) minus a small ATR cushion.
"""
from __future__ import annotations

from datetime import datetime

from ..models import FamilyScores, SetupCandidate, SetupType, Side
from ..session import is_opening_drive
from .base import Strategy, SymbolContext, clamp01

EARLY_RANGE_BARS = 5  # first 5 one-minute bars define the launch range


class GapAndGo(Strategy):
    name = "gap_and_go"

    def evaluate(self, ctx: SymbolContext, now: datetime) -> SetupCandidate | None:
        if not is_opening_drive(now, window_minutes=60):
            return None
        if len(ctx.bars_1m) <= EARLY_RANGE_BARS:
            return None
        price = ctx.last_price
        vwap = ctx.vwap.value
        atr_1m = ctx.atr_now()
        if price is None or vwap is None or atr_1m is None:
            return None

        side = Side.LONG if ctx.gap_pct > 0 else Side.SHORT  # gap sign decides side

        early = ctx.bars_1m[:EARLY_RANGE_BARS]
        early_high = max(b.high for b in early)
        early_low = min(b.low for b in early)

        imb = ctx.flow.taker_imbalance()
        slope = ctx.flow.cvd_slope(minutes=5)
        reasons: list[str] = [f"gap {ctx.gap_pct:+.1f}%"]

        if side == Side.LONG:
            if price <= vwap:
                return None                       # wrong side of VWAP: no trade
            if price <= early_high:
                return None                       # no breakout yet
            entry = price
            stop = max(vwap, early_low) - 0.25 * atr_1m
            position = clamp01((price - vwap) / (2.0 * atr_1m) + 0.5)
            flow_score = clamp01(0.5 * clamp01(imb / 0.3)            # imbalance vs +30%
                                 + 0.5 * (1.0 if (slope or 0) > 0 and ctx.flow.cvd > 0
                                          else 0.0))
            structure = clamp01((price - early_high) / max(atr_1m, 1e-9))
            reasons += [f"above VWAP {vwap:.2f}", f"broke 5m high {early_high:.2f}",
                        f"taker imb {imb:+.2f}", f"CVD {ctx.flow.cvd:+.0f}"]
        else:
            if price >= vwap or price >= early_low:
                return None
            entry = price
            stop = min(vwap, early_high) + 0.25 * atr_1m
            position = clamp01((vwap - price) / (2.0 * atr_1m) + 0.5)
            flow_score = clamp01(0.5 * clamp01(-imb / 0.3)
                                 + 0.5 * (1.0 if (slope or 0) < 0 and ctx.flow.cvd < 0
                                          else 0.0))
            structure = clamp01((early_low - price) / max(atr_1m, 1e-9))
            reasons += [f"below VWAP {vwap:.2f}", f"broke 5m low {early_low:.2f}",
                        f"taker imb {imb:+.2f}", f"CVD {ctx.flow.cvd:+.0f}"]

        if abs(entry - stop) < 1e-9:
            return None

        volatility = clamp01(abs(ctx.gap_pct) / 8.0)   # 8%+ gap = full marks
        scores = FamilyScores(position=position, orderflow=flow_score,
                              structure=structure, volatility=volatility)
        return SetupCandidate(symbol=ctx.symbol, setup=SetupType.GAP_AND_GO,
                              side=side, entry=entry, stop=stop,
                              scores=scores, reasons=reasons)
