"""Resolve pending predictions once a new bar has closed after the prediction."""
from __future__ import annotations

import logging
from datetime import datetime

from ..config import Config
from ..models import Bar, Direction
from ..technical.data import fetch_bars
from .recorder import PredictionRecorder

log = logging.getLogger(__name__)

# Moves smaller than this (in %) count as NEUTRAL, scaled to how much a given
# timeframe typically moves — a 30-minute bar and a 1-month bar shouldn't be
# graded against the same fixed threshold.
_FLAT_THRESHOLD_PCT_BY_TIMEFRAME: dict[str, float] = {
    "30m": 0.10,
    "1h": 0.10,
    "1d": 0.30,
    "1wk": 1.00,
    "1mo": 2.00,
}
_DEFAULT_FLAT_THRESHOLD_PCT = 0.30


def _actual_direction(prev_price: float, next_price: float, timeframe: str) -> Direction:
    pct = (next_price - prev_price) / prev_price * 100.0
    threshold = _FLAT_THRESHOLD_PCT_BY_TIMEFRAME.get(timeframe, _DEFAULT_FLAT_THRESHOLD_PCT)
    if abs(pct) < threshold:
        return Direction.NEUTRAL
    return Direction.UP if pct > 0 else Direction.DOWN


def _first_bar_after(bars: list[Bar], cutoff: datetime) -> Bar | None:
    for bar in bars:
        if bar.ts > cutoff:
            return bar
    return None


def run(cfg: Config, user_id: int | None = None) -> None:
    recorder = PredictionRecorder(cfg.db_path)
    try:
        rows = recorder.unresolved(user_id=user_id)
        for row in rows:
            bars = fetch_bars(row["symbol"], cfg, row["timeframe"])
            if not bars:
                continue
            pred_ts = datetime.fromisoformat(row["ts"])
            next_bar = _first_bar_after(bars, pred_ts)
            if next_bar is None:
                # No new bar has closed since the prediction was made yet —
                # resolving against "the latest price" here would just be
                # comparing the prediction to itself and always grade NEUTRAL.
                continue
            actual = _actual_direction(row["price_at_prediction"], next_bar.close, row["timeframe"])
            predicted = Direction(row["final_direction"])
            hit = actual == predicted
            recorder.resolve(row["id"], next_bar.close, actual, hit)
            log.info("backfill: %s (%s) predicted=%s actual=%s hit=%s",
                      row["symbol"], row["timeframe"], predicted.value, actual.value, hit)
    finally:
        recorder.close()
