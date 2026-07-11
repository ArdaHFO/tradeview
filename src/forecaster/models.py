"""Core data models shared across the forecaster pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass(frozen=True)
class Bar:
    """One daily OHLCV bar."""
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class NewsArticle:
    title: str
    source: str
    url: str
    published_ts: datetime
    snippet: str = ""


@dataclass
class NewsVerdict:
    symbol: str
    direction: Direction
    score: float                       # -1.0 .. 1.0
    confidence: float                  # 0.0 .. 1.0
    key_drivers: list[str] = field(default_factory=list)
    rationale: str = ""
    article_count: int = 0


@dataclass
class TechnicalVerdict:
    symbol: str
    score: float                       # -1.0 .. 1.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    ts: datetime
    symbol: str
    news_score: float
    news_confidence: float
    news_rationale: str
    technical_score: float
    technical_reasons: list[str]
    final_score: float
    final_direction: Direction
    final_confidence: float
    price_at_prediction: float
    actual_next_close: float | None = None
    actual_direction: Direction | None = None
    hit: bool | None = None
