"""Indicative entry/stop/target levels from EOD data (the 'day-2 playbook').

These are PLAN levels, not orders: long bias = break of yesterday's high,
short bias = break of yesterday's low, stop at 30% of yesterday's range,
target at 2R. Live CVD/VWAP confirmation is what turns a level into a signal.
"""
from __future__ import annotations

from dataclasses import dataclass

from .models import SymbolSnapshot


@dataclass(frozen=True)
class PlaybookLevels:
    side: str          # "LONG" | "SHORT"
    entry: float
    stop: float
    target: float


def playbook_levels(s: SymbolSnapshot) -> PlaybookLevels | None:
    rng = s.day_high - s.day_low
    if rng <= 0 or s.price <= 0:
        return None
    if s.gap_pct >= 0:
        entry = s.day_high * 1.005          # break of yesterday's high + buffer
        stop = entry - 0.30 * rng
        target = entry + 2.0 * (entry - stop)
        side = "LONG"
    else:
        entry = s.day_low * 0.995           # breakdown of yesterday's low
        stop = entry + 0.30 * rng
        target = entry - 2.0 * (stop - entry)
        side = "SHORT"
    return PlaybookLevels(side=side, entry=round(entry, 2), stop=round(stop, 2),
                          target=round(target, 2))
