"""Open-position tracking: the SELL half of the system.

A scanner that only says what to buy is half a strategy. Every backtested result
in this repo depends on positions being closed by a specific rule -- Connors
RSI(2) exits when price reclaims the 5-day mean, not when the holder feels
finished -- so the exit has to be as mechanical and as visible as the entry.

You record what you actually filled; this reads the latest daily bar and returns
SELL or HOLD per position, using the same rules the backtest used. Three ways
out, checked in the order the backtest checks them:

  1. stop      -- price at or below the initial stop
  2. signal    -- the strategy's managed-exit condition (e.g. close > SMA5)
  3. time      -- held longer than the strategy's maximum

Discretionary exits are how a validated edge quietly becomes an unvalidated one:
cutting winners early and holding losers turns a positive expectancy negative
without changing a single entry.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .backtest import _exit_signal
from .strategies import Strategy, add_indicators

POSITIONS_FILE = Path("positions.json")


@dataclass
class Position:
    """A fill the user actually took."""
    symbol: str
    strategy: str
    entry_date: str            # YYYY-MM-DD
    entry_price: float
    shares: int
    stop: float


@dataclass
class PositionStatus:
    position: Position
    last_close: float
    last_date: str
    bars_held: int
    action: str                # SELL | HOLD
    reason: str
    exit_level: float | None   # where the managed exit currently sits
    unrealized_pnl: float
    unrealized_r: float
    days_left: int             # sessions until the time stop


def load_positions(path: Path = POSITIONS_FILE) -> list[Position]:
    if not Path(path).exists():
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Position(**p) for p in raw]


def save_positions(positions: list[Position],
                   path: Path = POSITIONS_FILE) -> None:
    Path(path).write_text(
        json.dumps([asdict(p) for p in positions], indent=1), encoding="utf-8")


def exit_level(rule: str, row: pd.Series) -> float | None:
    """The price the managed exit is currently sitting at, for display."""
    key = {"sma5": "sma5", "sma20": "sma20", "dc_low20": "dc_low20"}.get(rule)
    if key is None:
        return None
    value = row.get(key)
    return None if value is None or pd.isna(value) else float(value)


def status(position: Position, df: pd.DataFrame,
           strategy: Strategy) -> PositionStatus | None:
    """Evaluate one position against the latest bar. None if data is missing."""
    ind = add_indicators(df)
    if ind.empty:
        return None
    entry_ts = pd.Timestamp(position.entry_date)
    # Sessions elapsed since entry, counted the way the backtest counts them.
    held = int((ind.index > entry_ts).sum())
    row = ind.iloc[-1]
    close = float(row["Close"])

    if close <= position.stop:
        action, reason = "SELL", f"stop tetiklendi ({position.stop:.2f})"
    elif _exit_signal(strategy.exit_rule, row):
        action, reason = "SELL", f"çıkış kuralı ({strategy.exit_rule})"
    elif held >= strategy.max_hold_days:
        action, reason = "SELL", f"süre doldu ({strategy.max_hold_days} seans)"
    else:
        action, reason = "HOLD", "kural tetiklenmedi"

    risk_ps = position.entry_price - position.stop
    pnl = (close - position.entry_price) * position.shares
    return PositionStatus(
        position=position, last_close=close,
        last_date=str(ind.index[-1].date()), bars_held=held,
        action=action, reason=reason,
        exit_level=exit_level(strategy.exit_rule, row),
        unrealized_pnl=round(pnl, 2),
        unrealized_r=round((close - position.entry_price) / risk_ps, 2)
        if risk_ps > 0 else 0.0,
        days_left=max(0, strategy.max_hold_days - held))


def review(positions: list[Position], frames: dict[str, pd.DataFrame],
           build_strategy) -> list[PositionStatus]:
    """Status for every tracked position, SELLs listed first."""
    out: list[PositionStatus] = []
    for pos in positions:
        df = frames.get(pos.symbol)
        if df is None or df.empty:
            continue
        st = status(pos, df, build_strategy(pos.strategy))
        if st is not None:
            out.append(st)
    out.sort(key=lambda s: (s.action != "SELL", -s.unrealized_r))
    return out


def format_review(rows: list[PositionStatus]) -> str:
    if not rows:
        return ("=== AÇIK POZİSYONLAR ===\n"
                "  (kayıtlı pozisyon yok)\n\n"
                "  Alım yaptığında kaydet:\n"
                "    python main.py swing-positions --add AAPL "
                "--shares 3 --price 308.91")
    sells = [r for r in rows if r.action == "SELL"]
    lines = ["=== AÇIK POZİSYONLAR ===",
             f"  {'SEMBOL':8s}{'GİRİŞ':>9s}{'SON':>9s}{'STOP':>9s}"
             f"{'ÇIKIŞ':>9s}{'K/Z':>10s}{'R':>7s}{'GÜN':>6s}  DURUM"]
    for r in rows:
        ex = f"{r.exit_level:.2f}" if r.exit_level is not None else "—"
        flag = "🔴 SAT" if r.action == "SELL" else "🟢 TUT"
        lines.append(
            f"  {r.position.symbol:8s}{r.position.entry_price:>9.2f}"
            f"{r.last_close:>9.2f}{r.position.stop:>9.2f}{ex:>9s}"
            f"{r.unrealized_pnl:>+10.2f}{r.unrealized_r:>+7.2f}"
            f"{r.bars_held:>6d}  {flag} — {r.reason}")
    total = sum(r.unrealized_pnl for r in rows)
    lines += ["  " + "─" * 78,
              f"  {len(rows)} pozisyon | gerçekleşmemiş K/Z ${total:+,.2f}"
              f" | satılacak: {len(sells)}"]
    if sells:
        lines.append("  ⚠ SAT işaretli pozisyonlar bir SONRAKİ seansın açılışında "
                     "kapatılır — backtest böyle çıktı aldı.")
    lines.append("  ⚠ Çıkış sütunu, kuralın şu an durduğu fiyat; sabit bir hedef "
                 "değil, her gün hareket eder.")
    return "\n".join(lines)
