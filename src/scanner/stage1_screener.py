"""Stage-1 universe screener: ~8000 symbols -> 10-30 'in play' watchlist.

Filters (defaults, all configurable in ScreenerConfig):
  RVOL >= 2, |gap| >= 4%, $2 <= price <= $50, float < 50M, day range >= 4%.
Unknown float passes by default with a score penalty (require_float=False).
"""
from __future__ import annotations

from .config import ScreenerConfig
from .features.indicators import rvol as rvol_calc
from .models import SymbolSnapshot, WatchlistItem


def screen(snapshots: list[SymbolSnapshot], cfg: ScreenerConfig,
           elapsed_fraction: float) -> list[WatchlistItem]:
    """Apply hard filters, rank by heat score, cap at max_watchlist_size."""
    items: list[WatchlistItem] = []
    for s in snapshots:
        if not (cfg.min_price <= s.price <= cfg.max_price):
            continue
        gap = s.gap_pct
        if abs(gap) < cfg.min_abs_gap_pct:
            continue
        rv = rvol_calc(s.day_volume, s.avg_daily_volume, elapsed_fraction)
        if rv < cfg.min_rvol:
            continue
        if s.day_range_pct < cfg.min_day_range_pct:
            continue
        if s.float_shares is not None and s.float_shares > cfg.max_float_shares:
            continue
        if s.float_shares is None and cfg.require_float:
            continue
        items.append(WatchlistItem(
            symbol=s.symbol,
            rvol=rv,
            gap_pct=gap,
            price=s.price,
            float_shares=s.float_shares,
            day_range_pct=s.day_range_pct,
            score=_heat_score(rv, gap, s),
        ))
    items.sort(key=lambda w: w.score, reverse=True)
    return items[:cfg.max_watchlist_size]


def _heat_score(rv: float, gap_pct: float, s: SymbolSnapshot) -> float:
    """Ranking heuristic: more relative volume, bigger gap, smaller float = hotter."""
    # RVOL dominates: full credit up to 10x, then diminishing (1pt per extra x)
    # so a 40x runner still outranks a 12x one instead of tying at the cap.
    score = min(rv, 10.0) * 10.0 + min(max(rv - 10.0, 0.0), 50.0)
    score += min(abs(gap_pct), 30.0) * 2.0  # gap contributes up to 60
    if s.float_shares is not None:
        if s.float_shares < 10_000_000:
            score += 25.0
        elif s.float_shares < 25_000_000:
            score += 15.0
        elif s.float_shares < 50_000_000:
            score += 5.0
    else:
        score -= 10.0                        # unknown float: mild penalty
    if s.has_news:
        score += 15.0                        # confirmed catalyst
    return score
