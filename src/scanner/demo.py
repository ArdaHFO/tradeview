"""Synthetic end-to-end demo: fabricates a gap-up momentum session and runs the
full pipeline (screener -> engine -> signals) without any API key.

Purpose: prove the plumbing works and show what a signal looks like.
The price path is scripted (gap up, hold VWAP, break the early high on heavy
buy flow), so the demo SHOULD produce at least one GAP_AND_GO signal.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import Config
from .engine import SignalEngine
from .models import Bar, Quote, SymbolSnapshot, TradeTick
from .session import NY
from .signal.recorder import SignalRecorder
from .stage1_screener import screen


def _ny(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=NY)


def build_demo_snapshots() -> list[SymbolSnapshot]:
    """A fake market: one hot gapper, a few names that must be filtered out."""
    return [
        # The play: +12% gap, 6x volume, 18M float, wide range
        SymbolSnapshot(symbol="HOTX", price=11.20, prev_close=10.00,
                       day_volume=9_000_000, avg_daily_volume=1_500_000,
                       day_high=11.60, day_low=10.80,
                       float_shares=18_000_000, has_news=True),
        # Too big a float and no gap -> filtered
        SymbolSnapshot(symbol="MEGA", price=150.0, prev_close=149.0,
                       day_volume=30_000_000, avg_daily_volume=40_000_000,
                       day_high=151.0, day_low=148.5,
                       float_shares=2_000_000_000),
        # Gap but no volume -> filtered
        SymbolSnapshot(symbol="THIN", price=5.40, prev_close=5.00,
                       day_volume=80_000, avg_daily_volume=900_000,
                       day_high=5.5, day_low=5.1, float_shares=8_000_000),
        # Penny -> filtered by price floor
        SymbolSnapshot(symbol="PNNY", price=0.80, prev_close=0.60,
                       day_volume=20_000_000, avg_daily_volume=2_000_000,
                       day_high=0.95, day_low=0.62, float_shares=30_000_000),
    ]


def run_demo(cfg: Config, db_path: str = ":memory:") -> tuple[list, list]:
    """Returns (watchlist, signals)."""
    rng = random.Random(42)

    # ---- stage 1 ----
    snapshots = build_demo_snapshots()
    watchlist = screen(snapshots, cfg.screener, elapsed_fraction=0.08)

    # ---- stage 2: scripted session for the surviving symbol(s) ----
    recorder = SignalRecorder(db_path)
    engine = SignalEngine(cfg, watchlist, recorder=recorder)

    day = _ny(2026, 6, 11, 9, 30)
    price = 11.00
    for minute in range(45):                      # 09:30 - 10:15
        ts = day + timedelta(minutes=minute)
        # Scripted drift: shallow pullback minutes 5-9, breakout from minute 10.
        if minute < 5:
            drift = 0.02
        elif minute < 10:
            drift = -0.015
        else:
            drift = 0.05
        o = price
        c = max(0.5, price + drift + rng.uniform(-0.01, 0.01))
        h = max(o, c) + rng.uniform(0.0, 0.02)
        low = min(o, c) - rng.uniform(0.0, 0.02)
        vol = rng.uniform(80_000, 140_000) * (2.0 if minute >= 10 else 1.0)

        # Quotes + trades inside the bar: buy-heavy after the pullback.
        bid, ask = c - 0.01, c + 0.01
        engine.on_quote("HOTX", Quote(ts=ts, bid=bid, ask=ask))
        n_trades = 30
        buy_ratio = 0.75 if minute >= 8 else 0.55
        for k in range(n_trades):
            tts = ts + timedelta(seconds=k * 2)
            if rng.random() < buy_ratio:
                engine.on_trade("HOTX", TradeTick(ts=tts, price=ask,
                                                  size=rng.uniform(200, 800)))
            else:
                engine.on_trade("HOTX", TradeTick(ts=tts, price=bid,
                                                  size=rng.uniform(100, 500)))

        engine.on_bar("HOTX", Bar(ts=ts, open=o, high=h, low=low, close=c,
                                  volume=vol))
        price = c

    recorder.close()
    return watchlist, engine.signals
