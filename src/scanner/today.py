"""'today' mode: replay TODAY's session from free Yahoo data.

After (or during) the session, pull today's movers, run the signal engine over
today's real 1m bars, and report every signal with entry/stop/target and the
simulated outcome so far. Also records signals to the DB -> dashboard shows them.

'live-yf' mode: same plumbing, but polls for new bars every minute while the
market is open and alerts in near-real-time (~1-2 min Yahoo lag).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .alerts.telegram import TelegramAlerter
from .backtest import Backtester, TradeResult
from .config import Config
from .data.yahoo import fetch_1m_bars, today_movers
from .engine import SignalEngine, feed_synthetic_flow
from .models import Bar, Signal, WatchlistItem
from .session import is_rth, minutes_to_close, session_elapsed_fraction
from .signal.recorder import SignalRecorder
from .stage1_screener import screen

log = logging.getLogger(__name__)


def build_today_watchlist(cfg: Config) -> list[WatchlistItem]:
    snapshots = today_movers()
    elapsed = session_elapsed_fraction(datetime.now(timezone.utc))
    return screen(snapshots, cfg.screener, elapsed_fraction=max(elapsed, 0.2))


def replay_today(cfg: Config, top_n: int = 10
                 ) -> tuple[list[WatchlistItem], list[TradeResult]]:
    """Run the engine over today's bars; returns (watchlist, simulated trades)."""
    watchlist = build_today_watchlist(cfg)[:top_n]
    if not watchlist:
        return [], []
    bars_by_sym = fetch_1m_bars([w.symbol for w in watchlist])
    bt = Backtester(cfg)                     # reuse its fill simulator
    recorder = SignalRecorder(cfg.db_path)
    results: list[TradeResult] = []
    for w in watchlist:
        bars = [b for b in bars_by_sym.get(w.symbol, []) if is_rth(b.ts)]
        if len(bars) < 10:
            continue
        engine = SignalEngine(cfg, [w], recorder=recorder)
        pending: list[tuple[Signal, int]] = []
        for i, bar in enumerate(bars):
            feed_synthetic_flow(engine, w.symbol, bar)
            for sig in engine.on_bar(w.symbol, bar):
                pending.append((sig, i))
        for sig, i in pending:
            results.append(bt._simulate(sig, bars[i + 1:]))
    recorder.close()
    return watchlist, results


def run_live_yahoo(cfg: Config, top_n: int = 15, poll_sec: int = 60) -> None:
    """Poll Yahoo every minute during RTH and emit signals as bars arrive."""
    watchlist = build_today_watchlist(cfg)[:top_n]
    if not watchlist:
        print("Şu an filtreyi geçen hisse yok (piyasa kapalı olabilir).")
        return
    symbols = [w.symbol for w in watchlist]
    print(f"Canlı izleme ({len(symbols)}): {', '.join(symbols)}")
    alerter = TelegramAlerter(cfg.telegram_bot_token, cfg.telegram_chat_id)
    recorder = SignalRecorder(cfg.db_path)
    engine = SignalEngine(cfg, watchlist, alerter=alerter, recorder=recorder)
    last_ts: dict[str, datetime] = {}
    try:
        while True:
            now = datetime.now(timezone.utc)
            if not is_rth(now):
                print("Piyasa kapalı — canlı mod durdu.")
                break
            if minutes_to_close(now) <= 1:
                print("Kapanışa 1 dk — canlı mod durdu.")
                break
            bars_by_sym = fetch_1m_bars(symbols)
            for sym, bars in bars_by_sym.items():
                cutoff = last_ts.get(sym)
                fresh: list[Bar] = [b for b in bars if is_rth(b.ts)
                                    and (cutoff is None or b.ts > cutoff)]
                for bar in fresh:
                    feed_synthetic_flow(engine, sym, bar)
                    engine.on_bar(sym, bar)
                if fresh:
                    last_ts[sym] = fresh[-1].ts
            time.sleep(poll_sec)
    finally:
        recorder.close()
