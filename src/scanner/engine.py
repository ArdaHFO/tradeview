"""Stage-2 signal engine: routes market data into SymbolContexts, evaluates
strategies on each completed 1m bar, gates through risk, scores confluence,
records and alerts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from .alerts.telegram import TelegramAlerter
from .config import Config
from .models import Bar, Quote, SetupCandidate, Signal, TradeTick, WatchlistItem
from .risk import RiskManager
from .signal.recorder import SignalRecorder
from .signal.scorer import ConfluenceScorer
from .strategies import ALL_STRATEGIES, Strategy, SymbolContext

log = logging.getLogger(__name__)


def feed_synthetic_flow(engine: "SignalEngine", symbol: str, bar: Bar) -> None:
    """Approximate intra-bar order flow from a bar's shape (no tick data).

    buy_ratio = (close - low) / (high - low): a close near the high means
    buyers were lifting offers. Two synthetic prints (one buy at ask, one
    sell at bid) preserve the volume split for the CVD/imbalance trackers.
    Used by the backtester and the Yahoo polling mode; real tick feeds
    (paid Polygon) call on_trade/on_quote directly instead.
    """
    rng = bar.high - bar.low
    buy_ratio = 0.5 if rng <= 0 else (bar.close - bar.low) / rng
    spread = max(bar.close * 0.0005, 0.01)
    bid, ask = bar.close - spread, bar.close + spread
    engine.on_quote(symbol, Quote(ts=bar.ts, bid=bid, ask=ask))
    buy_vol = bar.volume * buy_ratio
    sell_vol = bar.volume * (1.0 - buy_ratio)
    if buy_vol > 0:
        engine.on_trade(symbol, TradeTick(ts=bar.ts, price=ask, size=buy_vol))
    if sell_vol > 0:
        engine.on_trade(symbol, TradeTick(ts=bar.ts, price=bid, size=sell_vol))


class SignalEngine:
    def __init__(self, cfg: Config, watchlist: list[WatchlistItem],
                 alerter: TelegramAlerter | None = None,
                 recorder: SignalRecorder | None = None) -> None:
        self.cfg = cfg
        self.contexts: dict[str, SymbolContext] = {
            w.symbol: SymbolContext(symbol=w.symbol, gap_pct=w.gap_pct,
                                    cfg=cfg.signal)
            for w in watchlist
        }
        disabled = {s.strip().upper() for s in cfg.signal.disabled_setups.split(",")
                    if s.strip()}
        self.strategies: list[Strategy] = [
            S(cfg.signal) for S in ALL_STRATEGIES
            if S.name.upper() not in disabled
            and S.name.upper().replace("_", "") not in disabled]
        self.cooldown = timedelta(minutes=cfg.signal.cooldown_minutes)
        self.scorer = ConfluenceScorer(cfg.signal)
        self.risk = RiskManager(cfg.risk)
        self.alerter = alerter
        self.recorder = recorder
        self.signals: list[Signal] = []
        self._last_alert: dict[tuple[str, str], datetime] = {}

    # ---- data routing ---------------------------------------------------

    def on_trade(self, symbol: str, trade: TradeTick) -> None:
        ctx = self.contexts.get(symbol)
        if ctx:
            ctx.on_trade(trade)

    def on_quote(self, symbol: str, quote: Quote) -> None:
        ctx = self.contexts.get(symbol)
        if ctx:
            ctx.on_quote(quote)

    def on_bar(self, symbol: str, bar: Bar) -> list[Signal]:
        """Bar close is the evaluation trigger. Returns any signals emitted."""
        ctx = self.contexts.get(symbol)
        if ctx is None:
            return []
        ctx.on_bar(bar)
        now = bar.ts
        self.risk.on_new_session(now)
        return self._evaluate(ctx, now)

    # ---- evaluation -----------------------------------------------------

    def _evaluate(self, ctx: SymbolContext, now: datetime) -> list[Signal]:
        allowed, why = self.risk.can_enter(now)
        if not allowed:
            log.debug("%s entry blocked: %s", ctx.symbol, why)
            return []
        emitted: list[Signal] = []
        for strat in self.strategies:
            cand = strat.evaluate(ctx, now)
            if cand is None:
                continue
            if self._in_cooldown(cand, now):
                continue
            # Cost filter: if the stop is so tight that round-trip slippage eats
            # a large share of R, the trade is structurally unprofitable -- skip.
            one_side_slip = cand.entry * self.cfg.signal.slippage_bps / 10_000.0
            if abs(cand.entry - cand.stop) < \
                    self.cfg.signal.min_stop_slippage_mult * one_side_slip:
                continue
            shares, risk_dollars = self.risk.position_size(cand.entry, cand.stop)
            if shares <= 0:
                continue
            sig = self.scorer.score(cand, now, shares, risk_dollars)
            if sig is None:
                continue
            self._last_alert[(cand.symbol, cand.setup.value)] = now
            self.signals.append(sig)
            emitted.append(sig)
            if self.recorder:
                self.recorder.record(sig)
            if self.alerter:
                self.alerter.send(sig)
            log.info("signal: %s %s %s @ %.2f (conf %.2f)",
                     sig.symbol, sig.setup.value, sig.side.value,
                     sig.entry, sig.confluence)
        return emitted

    def _in_cooldown(self, cand: SetupCandidate, now: datetime) -> bool:
        last = self._last_alert.get((cand.symbol, cand.setup.value))
        return last is not None and now - last < self.cooldown
