"""Today's swing entries: run a validated strategy over the latest daily bars.

This is the live end of the swing pipeline. It applies exactly the strategy and
sizing the backtest used, so what it prints is what was measured -- no separate
"live" code path to drift out of sync.

Timing matters and is easy to get wrong: signals are computed from the last
CLOSED daily bar and are meant to be filled at the NEXT session's open, which is
how the backtest fills them. Acting on an intraday price instead changes the
strategy into one that was never tested.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .backtest import SwingConfig
from .positions import exit_level
from .strategies import Strategy, add_indicators


@dataclass
class Candidate:
    symbol: str
    signal_date: pd.Timestamp
    close: float
    atr: float
    stop: float
    risk_per_share: float
    shares: int
    notional: float
    rank: float
    reason: str
    exit_level: float | None    # where the managed exit currently sits
    exit_rule: str              # what closes the trade
    max_hold_days: int          # time stop


def scan_today(frames: dict[str, pd.DataFrame], strategy: Strategy,
               cfg: SwingConfig | None = None,
               as_of: pd.Timestamp | None = None) -> tuple[list[Candidate],
                                                           pd.Timestamp | None]:
    """Entries triggered on the most recent closed bar, sized per `cfg`.

    Returns (candidates, bar_date). Candidates are ranked by the strategy's own
    ranking and capped at `max_positions`, mirroring how the backtest allocates
    scarce slots.
    """
    cfg = cfg or SwingConfig()
    latest: pd.Timestamp | None = as_of
    if latest is None:
        for df in frames.values():
            if len(df.index):
                last = df.index[-1]
                latest = last if latest is None else max(latest, last)
    if latest is None:
        return [], None

    out: list[Candidate] = []
    for sym, df in frames.items():
        if latest not in df.index:
            continue                       # no bar for this symbol on that date
        ind = add_indicators(df)
        ent = strategy.entries(ind)
        if latest not in ent.index:
            continue
        row = ind.loc[latest]
        if not (row["dollar_vol20"] >= cfg.min_dollar_volume):
            continue
        close = float(row["Close"])
        atr_val = float(row["atr14"])
        if not (atr_val > 0) or not (close > 0):
            continue
        # Sized off the last close as a planning estimate; the real fill is the
        # next open, so treat share count as approximate.
        stop = close - strategy.stop_atr * atr_val
        risk_ps = close - stop
        shares = int((cfg.equity * cfg.risk_pct / 100.0) / risk_ps)
        cap = int((cfg.equity * cfg.max_position_pct / 100.0) / close)
        shares = min(shares, cap)
        if shares < 1:
            continue
        out.append(Candidate(symbol=sym, signal_date=latest, close=close,
                             atr=atr_val, stop=stop, risk_per_share=risk_ps,
                             shares=shares, notional=shares * close,
                             rank=float(ent.loc[latest, "rank"]),
                             reason=str(ent.loc[latest, "reason"]),
                             exit_level=exit_level(strategy.exit_rule, row),
                             exit_rule=strategy.exit_rule,
                             max_hold_days=strategy.max_hold_days))
    out.sort(key=lambda c: -c.rank)
    return out[:cfg.max_positions], latest


def format_scan(cands: list[Candidate], bar_date: pd.Timestamp | None,
                strategy_name: str, cfg: SwingConfig) -> str:
    lines = [f"=== SWING SİNYALLERİ — {strategy_name} ===",
             f"  Son kapanış barı : {bar_date.date() if bar_date is not None else '—'}",
             f"  Sermaye          : ${cfg.equity:,.0f}"
             f" | işlem başı risk %{cfg.risk_pct:g}",
             ""]
    if not cands:
        lines.append("  (bugün sinyal yok)")
        return "\n".join(lines)
    first = cands[0]
    lines.append(f"  {'SEMBOL':8s}{'AL~':>9s}{'STOP':>9s}{'SAT~':>9s}"
                 f"{'ADET':>6s}{'TUTAR':>10s}{'RİSK':>8s}  GEREKÇE")
    for c in cands:
        ex = f"{c.exit_level:.2f}" if c.exit_level is not None else "—"
        lines.append(f"  {c.symbol:8s}{c.close:>9.2f}{c.stop:>9.2f}{ex:>9s}"
                     f"{c.shares:>6d}{c.notional:>10,.0f}"
                     f"{c.shares * c.risk_per_share:>8,.0f}  {c.reason}")
    lines += [
        "",
        f"  ÇIKIŞ KURALI ({first.exit_rule}) — üç yoldan biriyle kapanır:",
        "    1) STOP  : fiyat stop seviyesine düşerse (zarar kes)",
        f"    2) SAT~  : kapanış '{first.exit_rule}' seviyesini geçerse (kâr al)",
        f"    3) SÜRE  : {first.max_hold_days} seans dolarsa (ne olursa olsun çık)",
        "",
        "  ⚠ SAT~ sütunu SABİT bir hedef DEĞİL — o seviye her gün hareket eder.",
        "    Pozisyonu aldıktan sonra günlük takip için:",
        "      python main.py swing-positions",
        "  ⚠ Giriş ve çıkış bir SONRAKİ seansın AÇILIŞINDA yapılır — backtest böyle",
        "    doldurdu. Gün içi fiyattan işlem test edilmemiş bir strateji olur.",
        "  ⚠ Adet, son kapanışa göre tahmindir; gerçek fill açılış fiyatıdır."]
    return "\n".join(lines)
