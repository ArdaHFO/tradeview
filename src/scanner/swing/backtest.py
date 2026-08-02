"""Portfolio backtest for daily-bar swing strategies.

Fill and accounting rules, chosen to err against the strategy rather than for it:

  * A signal computed on the close of day D is filled at the OPEN of D+1. Never
    at D's close -- that price was not knowable when the signal formed.
  * Within a day, if both the stop and the exit condition could have triggered,
    the STOP is taken. Daily bars cannot resolve intraday sequence, so the
    pessimistic branch is assumed.
  * A gap through the stop fills at the open, not at the stop price. Gap risk is
    the main way daily-bar backtests flatter themselves.
  * Slippage is charged on entry and exit, plus commission per side.
  * Position sizing is risk-based: risk_pct of equity divided by the stop
    distance. Capped by `max_position_pct` so one tight stop cannot swallow the
    book.
  * Slots are finite (`max_positions`). When more signals fire than slots exist,
    the strategy's own `rank` decides, which is part of the strategy.

Output is `validation.Trade`, so swing runs go through exactly the same
bootstrap/Monte Carlo scrutiny as the intraday ones.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from ..validation import Trade
from .strategies import Strategy, add_indicators

log = logging.getLogger(__name__)


@dataclass
class SwingConfig:
    equity: float = 10_000.0
    risk_pct: float = 1.0            # equity fraction risked per position
    max_positions: int = 10
    max_position_pct: float = 20.0   # cap on notional per position
    slippage_bps: float = 5.0        # per side; daily liquid names are tight
    commission: float = 1.0          # per side, flat
    min_dollar_volume: float = 5_000_000.0   # liquidity floor at signal time


@dataclass
class OpenPosition:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    stop: float
    risk_per_share: float
    bars_held: int = 0
    reason: str = ""


@dataclass
class SwingReport:
    strategy: str
    trades: list[Trade]
    equity_curve: pd.Series
    n_symbols: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    skipped_no_slot: int = 0
    warnings: list[str] = field(default_factory=list)


def _exit_signal(rule: str, row: pd.Series) -> bool:
    """Has the strategy's managed-exit condition triggered on this bar's close?"""
    if rule == "sma5":
        return bool(row["Close"] > row["sma5"])
    if rule == "sma20":
        return bool(row["Close"] < row["sma20"])
    if rule == "dc_low20":
        return bool(row["Close"] < row["dc_low20"])
    return False


def run(frames: dict[str, pd.DataFrame], strategy: Strategy,
        cfg: SwingConfig | None = None, progress=None) -> SwingReport:
    cfg = cfg or SwingConfig()
    if progress:
        progress(f"{len(frames)} sembol için gösterge hesaplanıyor")

    data = {sym: add_indicators(df) for sym, df in frames.items()}

    # Collect every entry signal up front, then simulate the portfolio over time.
    if progress:
        progress("sinyaller taranıyor")
    pending: dict[pd.Timestamp, list[tuple[float, str, str]]] = {}
    for sym, df in data.items():
        ent = strategy.entries(df)
        if ent.empty:
            continue
        liquid = df["dollar_vol20"].reindex(ent.index)
        for ts, row in ent.iterrows():
            if not (liquid.loc[ts] >= cfg.min_dollar_volume):
                continue                      # too thin to trade at size
            pending.setdefault(ts, []).append((float(row["rank"]), sym,
                                               str(row["reason"])))

    calendar = pd.DatetimeIndex(sorted(set().union(*(df.index for df in data.values()))))
    equity = cfg.equity
    open_pos: dict[str, OpenPosition] = {}
    trades: list[Trade] = []
    curve: list[float] = []
    queued: list[tuple[float, str, str]] = []      # signals awaiting next open
    skipped = 0

    if progress:
        progress(f"{len(calendar)} seans simüle ediliyor")

    for today in calendar:
        # 1) manage open positions on today's bar (before new entries compete)
        for sym in list(open_pos):
            pos = open_pos[sym]
            df = data[sym]
            if today not in df.index:
                continue
            row = df.loc[today]
            pos.bars_held += 1
            exit_price: float | None = None
            reason = ""
            if row["Open"] <= pos.stop:            # gapped through the stop
                exit_price, reason = float(row["Open"]), "gap_stop"
            elif row["Low"] <= pos.stop:           # pessimistic: stop before target
                exit_price, reason = pos.stop, "stop"
            elif _exit_signal(strategy.exit_rule, row):
                exit_price, reason = float(row["Close"]), "signal"
            elif pos.bars_held >= strategy.max_hold_days:
                exit_price, reason = float(row["Close"]), "time"
            if exit_price is None:
                continue
            slip = exit_price * cfg.slippage_bps / 10_000.0
            fill = exit_price - slip
            gross = (fill - pos.entry_price) * pos.shares
            net = gross - cfg.commission
            equity += net
            trades.append(Trade(
                day=str(today.date()), time="16:00", symbol=sym,
                setup=strategy.name.upper(), side="LONG",
                entry=round(pos.entry_price, 4), exit=round(fill, 4),
                exit_reason=reason,
                r_multiple=round(net / (pos.risk_per_share * pos.shares), 3)
                if pos.risk_per_share * pos.shares > 0 else 0.0,
                pnl=round(net, 2)))
            del open_pos[sym]

        # 2) fill yesterday's signals at today's open
        for rank, sym, why in sorted(queued, key=lambda x: -x[0]):
            if len(open_pos) >= cfg.max_positions:
                skipped += 1
                continue
            if sym in open_pos:
                continue
            df = data[sym]
            if today not in df.index:
                continue
            row = df.loc[today]
            atr_val = float(row["atr14"])
            open_px = float(row["Open"])
            if not (atr_val > 0) or not (open_px > 0):
                continue
            entry = open_px + open_px * cfg.slippage_bps / 10_000.0
            stop = entry - strategy.stop_atr * atr_val
            risk_ps = entry - stop
            if risk_ps <= 0:
                continue
            shares = int((equity * cfg.risk_pct / 100.0) / risk_ps)
            cap = int((equity * cfg.max_position_pct / 100.0) / entry)
            shares = min(shares, cap)
            if shares < 1:
                continue
            equity -= cfg.commission
            open_pos[sym] = OpenPosition(symbol=sym, entry_date=today,
                                         entry_price=entry, shares=shares,
                                         stop=stop, risk_per_share=risk_ps,
                                         reason=why)
        queued = pending.get(today, [])       # today's signals fill tomorrow
        curve.append(equity)

    # close anything still open at the final bar, so the sample has no survivors
    for sym, pos in open_pos.items():
        df = data[sym]
        last = df.iloc[-1]
        fill = float(last["Close"])
        net = (fill - pos.entry_price) * pos.shares - cfg.commission
        equity += net
        trades.append(Trade(
            day=str(df.index[-1].date()), time="16:00", symbol=sym,
            setup=strategy.name.upper(), side="LONG",
            entry=round(pos.entry_price, 4), exit=round(fill, 4),
            exit_reason="eod", pnl=round(net, 2),
            r_multiple=round(net / (pos.risk_per_share * pos.shares), 3)
            if pos.risk_per_share * pos.shares > 0 else 0.0))

    trades.sort(key=lambda t: t.day)
    warnings = ["Evren bugünün S&P 500 listesi: survivorship bias var, "
                "gerçek sonuç bundan kötüdür."]
    if skipped:
        warnings.append(f"{skipped} sinyal slot dolu olduğu için alınamadı "
                        f"(max_positions={cfg.max_positions}).")
    return SwingReport(strategy=strategy.name, trades=trades,
                       equity_curve=pd.Series(curve, index=calendar),
                       n_symbols=len(data),
                       start=calendar[0] if len(calendar) else None,
                       end=calendar[-1] if len(calendar) else None,
                       skipped_no_slot=skipped, warnings=warnings)


def format_report(rep: SwingReport, cfg: SwingConfig,
                  benchmark: bool = True) -> str:
    """Summary plus, unless disabled, the buy-and-hold comparison that decides
    whether the strategy earns its complexity."""
    n = len(rep.trades)
    lines = [f"=== SWING BACKTEST — {rep.strategy} ===",
             f"  Evren    : {rep.n_symbols} sembol",
             f"  Dönem    : {rep.start.date() if rep.start is not None else '—'}"
             f" → {rep.end.date() if rep.end is not None else '—'}",
             f"  İşlem    : {n}"]
    if n:
        wins = sum(1 for t in rep.trades if t.pnl > 0)
        gross_w = sum(t.pnl for t in rep.trades if t.pnl > 0)
        gross_l = -sum(t.pnl for t in rep.trades if t.pnl < 0)
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        total = sum(t.pnl for t in rep.trades)
        final = rep.equity_curve.iloc[-1]
        ret = (final / cfg.equity - 1.0) * 100.0
        lines += [f"  Win rate : {wins / n * 100:.1f}%",
                  f"  PF       : {pf:.2f}",
                  f"  PnL      : ${total:+,.2f}",
                  f"  Sermaye  : ${cfg.equity:,.0f} → ${final:,.2f} ({ret:+.1f}%)"]
        by_reason: dict[str, int] = {}
        for t in rep.trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1
        lines.append("  Çıkışlar : " + ", ".join(f"{k} {v}" for k, v
                                                 in sorted(by_reason.items())))
    for w in rep.warnings:
        lines.append(f"  ⚠ {w}")
    if n and benchmark is not False:
        from .benchmark import compare, format_comparison
        lines += ["", format_comparison(compare(rep.equity_curve))]
    return "\n".join(lines)
