"""Core data models shared across the scanner pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SetupType(str, Enum):
    GAP_AND_GO = "GAP_AND_GO"
    VWAP_REVERSION = "VWAP_REVERSION"
    ORB = "ORB"


@dataclass(frozen=True)
class Bar:
    """One OHLCV bar (any timeframe). `ts` is bar open time, UTC."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class TradeTick:
    """A single executed trade (time & sales print)."""
    ts: datetime
    price: float
    size: float


@dataclass(frozen=True)
class Quote:
    """Top-of-book NBBO quote."""
    ts: datetime
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class SymbolSnapshot:
    """Stage-1 screener input: one row per symbol from the daily market snapshot."""
    symbol: str
    price: float                    # last trade price
    prev_close: float
    day_volume: float               # cumulative volume so far today
    avg_daily_volume: float         # trailing average full-day volume
    day_high: float
    day_low: float
    float_shares: float | None = None   # free float; None if unknown
    atr_14: float | None = None         # daily ATR(14); None if unknown
    has_news: bool = False

    @property
    def gap_pct(self) -> float:
        if self.prev_close <= 0:
            return 0.0
        return (self.price - self.prev_close) / self.prev_close * 100.0

    @property
    def day_range_pct(self) -> float:
        if self.prev_close <= 0:
            return 0.0
        return (self.day_high - self.day_low) / self.prev_close * 100.0


@dataclass
class WatchlistItem:
    """A symbol that survived stage-1, with the metrics that qualified it."""
    symbol: str
    rvol: float
    gap_pct: float
    price: float
    float_shares: float | None
    day_range_pct: float
    score: float                    # screener ranking score (higher = hotter)


@dataclass
class FamilyScores:
    """Per-family confluence scores, each in [0, 1]."""
    position: float = 0.0     # family 1: VWAP / EMA location & trend
    orderflow: float = 0.0    # family 2: CVD, taker imbalance, tape
    structure: float = 0.0    # family 3: opening range, divergence, momentum
    volatility: float = 0.0   # family 4: ATR adequacy / regime

    def as_dict(self) -> dict[str, float]:
        return {
            "position": self.position,
            "orderflow": self.orderflow,
            "structure": self.structure,
            "volatility": self.volatility,
        }


@dataclass
class SetupCandidate:
    """Raw output of a strategy before confluence scoring."""
    symbol: str
    setup: SetupType
    side: Side
    entry: float
    stop: float
    scores: FamilyScores
    reasons: list[str] = field(default_factory=list)


@dataclass
class Signal:
    """Final, scored, risk-sized trade signal."""
    ts: datetime
    symbol: str
    setup: SetupType
    side: Side
    entry: float
    stop: float
    target: float
    confluence: float          # weighted total score in [0, 1]
    scores: FamilyScores
    position_size: int         # shares
    risk_dollars: float
    reasons: list[str] = field(default_factory=list)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)
