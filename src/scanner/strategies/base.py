"""Shared per-symbol live state and the strategy interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from ..config import SignalConfig
from ..features.indicators import atr, ema, opening_range, rsi
from ..features.orderflow import OrderFlowTracker
from ..features.vwap import SessionVWAP
from ..models import Bar, Quote, SetupCandidate, TradeTick


@dataclass
class SymbolContext:
    """Everything a strategy needs to know about one symbol, updated bar by bar.

    The engine owns one of these per watchlist symbol and calls
    `on_bar` / `on_trade` / `on_quote` as data arrives. Strategies read it.
    """
    symbol: str
    gap_pct: float                       # from the screener at session start
    cfg: SignalConfig
    bars_1m: list[Bar] = field(default_factory=list)
    vwap: SessionVWAP = field(default_factory=SessionVWAP)
    flow: OrderFlowTracker = field(default_factory=OrderFlowTracker)
    daily_atr: float | None = None       # ATR(14) on daily bars, for sizing context

    def __post_init__(self) -> None:
        self.flow = OrderFlowTracker(
            large_print_multiple=self.cfg.large_print_multiple)

    # ---- data ingestion ------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        self.bars_1m.append(bar)
        self.vwap.update_bar(bar)

    def on_trade(self, trade: TradeTick) -> None:
        self.flow.on_trade(trade)

    def on_quote(self, quote: Quote) -> None:
        self.flow.on_quote(quote)

    # ---- derived features (computed lazily per evaluation) -------------

    @property
    def last_price(self) -> float | None:
        return self.bars_1m[-1].close if self.bars_1m else None

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars_1m]

    def ema_now(self, period: int) -> float | None:
        series = ema(self.closes, period)
        return series[-1] if series else None

    def rsi_now(self, period: int = 9) -> float | None:
        series = rsi(self.closes, period)
        return series[-1] if series else None

    def atr_now(self, period: int = 14) -> float | None:
        """Intraday ATR on 1m bars; falls back to a scaled daily ATR early on."""
        series = atr(self.bars_1m, period)
        if series and series[-1] is not None:
            return series[-1]
        if self.daily_atr is not None:
            return self.daily_atr / 20.0   # rough 1m-scale fallback pre-warmup
        return None

    def opening_range(self) -> tuple[float, float] | None:
        return opening_range(self.bars_1m, self.cfg.opening_range_minutes)

    def avg_1m_volume(self, lookback: int = 20) -> float | None:
        if not self.bars_1m:
            return None
        window = self.bars_1m[-lookback:]
        return sum(b.volume for b in window) / len(window)


class Strategy(ABC):
    """A setup template: inspect the context, maybe emit a candidate."""

    name: str = "base"

    def __init__(self, cfg: SignalConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def evaluate(self, ctx: SymbolContext, now: datetime) -> SetupCandidate | None:
        ...


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
