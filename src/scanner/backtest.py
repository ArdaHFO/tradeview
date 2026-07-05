"""Minute-bar backtester (free-tier friendly), bias-free:

1. Build the watchlist for trading day D using ONLY data through D-1.
2. Fetch D's 1-minute bars for the top-N watchlist symbols.
3. Replay the RTH session through the real SignalEngine. Order flow is
   approximated from minute bars (buy ratio = (close-low)/(high-low)),
   since the free tier has no tick/quote history. This blunts the CVD
   edge, so treat results as a LOWER-BOUND sanity check, not final proof.
4. Simulate fills: stop checked before target inside the same bar
   (conservative), open trades closed at the session's last price.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import requests

from .config import Config
from .data.free_tier import BASE, THROTTLE_SEC, FreeTierScanner
from .engine import SignalEngine, feed_synthetic_flow
from .models import Bar, Side, Signal, WatchlistItem
from .session import is_rth
from .stage1_screener import screen

log = logging.getLogger(__name__)


@dataclass
class TradeResult:
    signal: Signal
    exit_price: float
    exit_reason: str        # stop | target | eod
    r_multiple: float
    pnl: float


@dataclass
class BacktestReport:
    trading_day: date
    symbols: list[str]
    results: list[TradeResult]

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def win_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.pnl > 0) / self.n

    @property
    def total_pnl(self) -> float:
        return sum(r.pnl for r in self.results)

    @property
    def avg_r(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.r_multiple for r in self.results) / self.n

    @property
    def profit_factor(self) -> float:
        wins = sum(r.pnl for r in self.results if r.pnl > 0)
        losses = -sum(r.pnl for r in self.results if r.pnl < 0)
        if losses <= 0:
            return float("inf") if wins > 0 else 0.0
        return wins / losses


class Backtester:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cfg.risk.pdt_account = False     # count every signal in stats
        self.http = requests.Session()
        self._last_call = 0.0
        self._scanner = FreeTierScanner(cfg.polygon_api_key)  # shared grouped cache

    # ---- data -----------------------------------------------------------

    def _minute_bars(self, symbol: str, day: date) -> list[Bar]:
        for attempt in range(4):
            wait = THROTTLE_SEC - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.http.get(
                f"{BASE}/v2/aggs/ticker/{symbol}/range/1/minute/"
                f"{day.isoformat()}/{day.isoformat()}",
                params={"apiKey": self.cfg.polygon_api_key, "adjusted": "true",
                        "sort": "asc", "limit": 50000}, timeout=60)
            if resp.status_code == 429:        # rate limit: back off and retry
                time.sleep(15 * (attempt + 1))
                continue
            break
        resp.raise_for_status()
        bars = []
        for r in resp.json().get("results") or []:
            ts = datetime.fromtimestamp(r["t"] / 1000.0, tz=timezone.utc)
            bars.append(Bar(ts=ts, open=float(r["o"]), high=float(r["h"]),
                            low=float(r["l"]), close=float(r["c"]),
                            volume=float(r["v"])))
        return bars

    # ---- replay ---------------------------------------------------------

    def _replay_symbol(self, item: WatchlistItem, bars: list[Bar],
                       prev_close: float) -> list[tuple[Signal, list[Bar]]]:
        rth = [b for b in bars if is_rth(b.ts)]
        if len(rth) < 10:
            return []
        # Real opening gap for day D (not D-1's day change):
        item.gap_pct = (rth[0].open - prev_close) / prev_close * 100.0
        engine = SignalEngine(self.cfg, [item])
        emitted: list[tuple[Signal, list[Bar]]] = []
        for i, bar in enumerate(rth):
            feed_synthetic_flow(engine, item.symbol, bar)
            for sig in engine.on_bar(item.symbol, bar):
                emitted.append((sig, rth[i + 1:]))
        return emitted

    def _simulate(self, sig: Signal, later_bars: list[Bar]) -> TradeResult:
        risk = abs(sig.entry - sig.stop)
        exit_price, reason = (later_bars[-1].close, "eod") if later_bars \
            else (sig.entry, "eod")
        for b in later_bars:
            if sig.side == Side.LONG:
                if b.low <= sig.stop:                  # conservative: stop first
                    exit_price, reason = sig.stop, "stop"
                    break
                if b.high >= sig.target:
                    exit_price, reason = sig.target, "target"
                    break
            else:
                if b.high >= sig.stop:
                    exit_price, reason = sig.stop, "stop"
                    break
                if b.low <= sig.target:
                    exit_price, reason = sig.target, "target"
                    break
        direction = 1.0 if sig.side == Side.LONG else -1.0
        # Per-side slippage: pay the spread/impact on the way in AND out.
        slip = sig.entry * self.cfg.signal.slippage_bps / 10_000.0
        gross = direction * (exit_price - sig.entry)
        net = gross - 2.0 * slip
        r_mult = net / risk if risk > 0 else 0.0
        pnl = r_mult * sig.risk_dollars
        return TradeResult(signal=sig, exit_price=exit_price, exit_reason=reason,
                           r_multiple=round(r_mult, 2), pnl=round(pnl, 2))

    # ---- top level ------------------------------------------------------

    def run(self, trading_day: date, top_n: int = 10,
            progress=None) -> BacktestReport:
        def say(msg: str) -> None:
            log.info(msg)
            if progress:
                progress(msg)

        say(f"building bias-free watchlist (data through {trading_day - timedelta(days=1)})")
        _, snapshots = self._scanner.scan(progress=progress,
                                          end_date=trading_day - timedelta(days=1))
        snap_by_sym = {s.symbol: s for s in snapshots}
        watchlist = screen(snapshots, self.cfg.screener, elapsed_fraction=1.0)
        picks = watchlist[:top_n]
        say(f"watchlist top {len(picks)}: " + ", ".join(w.symbol for w in picks))

        results: list[TradeResult] = []
        for w in picks:
            say(f"replaying {w.symbol} on {trading_day}")
            try:
                bars = self._minute_bars(w.symbol, trading_day)
            except requests.RequestException as exc:
                log.warning("bars failed for %s: %s", w.symbol, exc)
                continue
            prev_close = snap_by_sym[w.symbol].price   # close of D-1
            for sig, later in self._replay_symbol(w, bars, prev_close):
                results.append(self._simulate(sig, later))
        return BacktestReport(trading_day=trading_day,
                              symbols=[w.symbol for w in picks], results=results)


    def run_multi(self, end_day: date, n_days: int, top_n: int = 10,
                  progress=None) -> list[BacktestReport]:
        """Backtest the last `n_days` trading sessions ending at `end_day`."""
        days: list[date] = []
        d = end_day
        while len(days) < n_days:
            if d.weekday() < 5:
                days.append(d)
            d -= timedelta(days=1)
        days.reverse()
        reports = []
        for day in days:
            reports.append(self.run(day, top_n=top_n, progress=progress))
        return reports


def format_multi_report(reports: list[BacktestReport]) -> str:
    all_results = [r for rep in reports for r in rep.results]
    lines = ["=== ÇOK GÜNLÜ BACKTEST ÖZETİ ==="]
    for rep in reports:
        pf = rep.profit_factor
        pf_s = f"{pf:.2f}" if pf != float("inf") else "inf"
        lines.append(f"  {rep.trading_day}  trades {rep.n:3d}  "
                     f"win {rep.win_rate*100:3.0f}%  PF {pf_s:>5s}  "
                     f"PnL ${rep.total_pnl:+9.2f}")
    n = len(all_results)
    if n:
        wins = sum(1 for r in all_results if r.pnl > 0)
        gross_w = sum(r.pnl for r in all_results if r.pnl > 0)
        gross_l = -sum(r.pnl for r in all_results if r.pnl < 0)
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        total = sum(r.pnl for r in all_results)
        avg_r = sum(r.r_multiple for r in all_results) / n
        lines += ["  " + "─" * 56,
                  f"  TOPLAM: {n} işlem | win {wins/n*100:.0f}% | avg R {avg_r:+.2f}"
                  f" | profit factor {pf:.2f} | PnL ${total:+,.2f}"]
        # per-setup breakdown
        by_setup: dict[str, list] = {}
        for r in all_results:
            by_setup.setdefault(r.signal.setup.value, []).append(r)
        for name, rs in sorted(by_setup.items()):
            w = sum(1 for x in rs if x.pnl > 0)
            p = sum(x.pnl for x in rs)
            lines.append(f"    {name:15s} {len(rs):3d} işlem  win {w/len(rs)*100:3.0f}%"
                         f"  PnL ${p:+9.2f}")
    return "\n".join(lines)


def format_report(rep: BacktestReport) -> str:
    lines = [f"=== BACKTEST {rep.trading_day} — {len(rep.symbols)} symbols, "
             f"{rep.n} trades ==="]
    for r in rep.results:
        s = r.signal
        lines.append(
            f"  {s.ts.strftime('%H:%M')} {s.symbol:6s} {s.setup.value:14s} "
            f"{s.side.value:5s} in {s.entry:8.2f} out {r.exit_price:8.2f} "
            f"({r.exit_reason:6s}) R {r.r_multiple:+5.2f}  PnL ${r.pnl:+8.2f}")
    if rep.n:
        lines.append(f"  ── win rate {rep.win_rate*100:.0f}%  avg R {rep.avg_r:+.2f}  "
                     f"profit factor {rep.profit_factor:.2f}  "
                     f"total PnL ${rep.total_pnl:+.2f}")
    else:
        lines.append("  (no trades fired)")
    return "\n".join(lines)
