"""Persist every prediction to SQLite for later accuracy tracking."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..models import Direction, Prediction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    news_score REAL NOT NULL,
    news_confidence REAL NOT NULL,
    news_rationale TEXT NOT NULL,
    technical_score REAL NOT NULL,
    technical_reasons_json TEXT NOT NULL,
    final_score REAL NOT NULL,
    final_direction TEXT NOT NULL,
    final_confidence REAL NOT NULL,
    price_at_prediction REAL NOT NULL,
    actual_next_close REAL,
    actual_direction TEXT,
    hit INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, ts);
"""


class PredictionRecorder:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, p: Prediction) -> int:
        cur = self._conn.execute(
            "INSERT INTO predictions (ts, symbol, news_score, news_confidence, news_rationale,"
            " technical_score, technical_reasons_json, final_score, final_direction,"
            " final_confidence, price_at_prediction)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (p.ts.isoformat(), p.symbol, p.news_score, p.news_confidence, p.news_rationale,
             p.technical_score, json.dumps(p.technical_reasons), p.final_score,
             p.final_direction.value, p.final_confidence, p.price_at_prediction),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def unresolved(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT id, ts, symbol, final_direction, price_at_prediction"
            " FROM predictions WHERE hit IS NULL"
        )
        return cur.fetchall()

    def resolve(self, prediction_id: int, actual_next_close: float,
                actual_direction: Direction, hit: bool) -> None:
        self._conn.execute(
            "UPDATE predictions SET actual_next_close = ?, actual_direction = ?, hit = ?"
            " WHERE id = ?",
            (actual_next_close, actual_direction.value, int(hit), prediction_id),
        )
        self._conn.commit()

    def recent(self, limit: int = 50) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT ts, symbol, final_score, final_direction, final_confidence,"
            " actual_direction, hit FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cur.fetchall()

    def hit_rate(self, days: int) -> tuple[int, int]:
        cur = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(hit), 0) FROM predictions"
            " WHERE hit IS NOT NULL AND ts >= datetime('now', ?)",
            (f"-{days} days",),
        )
        total, hits = cur.fetchone()
        return int(hits), int(total)

    def close(self) -> None:
        self._conn.close()
