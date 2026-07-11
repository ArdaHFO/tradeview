"""Resolve yesterday's unresolved predictions against today's latest close."""
from __future__ import annotations

import logging

from ..config import Config
from ..models import Direction
from ..technical.data import latest_close
from .recorder import PredictionRecorder

log = logging.getLogger(__name__)

_FLAT_THRESHOLD_PCT = 0.05  # moves smaller than this count as NEUTRAL


def _actual_direction(prev_price: float, next_price: float) -> Direction:
    pct = (next_price - prev_price) / prev_price * 100.0
    if abs(pct) < _FLAT_THRESHOLD_PCT:
        return Direction.NEUTRAL
    return Direction.UP if pct > 0 else Direction.DOWN


def run(cfg: Config) -> None:
    recorder = PredictionRecorder(cfg.db_path)
    try:
        rows = recorder.unresolved()
        for row in rows:
            price = latest_close(row["symbol"], cfg)
            if price is None:
                continue
            actual = _actual_direction(row["price_at_prediction"], price)
            predicted = Direction(row["final_direction"])
            hit = actual == predicted
            recorder.resolve(row["id"], price, actual, hit)
            log.info("backfill: %s predicted=%s actual=%s hit=%s",
                      row["symbol"], predicted.value, actual.value, hit)
    finally:
        recorder.close()
