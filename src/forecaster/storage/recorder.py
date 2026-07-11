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
    timeframe TEXT NOT NULL,
    profile TEXT NOT NULL,
    news_sources TEXT NOT NULL,
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

CREATE TABLE IF NOT EXISTS watchlist (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    sector TEXT,
    notes TEXT,
    sources TEXT DEFAULT 'google',
    timeframes TEXT DEFAULT '1d',
    profiles TEXT DEFAULT 'balanced'
);

CREATE TABLE IF NOT EXISTS comparison_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    profile TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    news_score REAL NOT NULL,
    technical_score REAL NOT NULL,
    final_score REAL NOT NULL,
    final_direction TEXT NOT NULL,
    final_confidence REAL NOT NULL,
    news_sources TEXT NOT NULL,
    hit INTEGER
);
"""


class PredictionRecorder:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, p: Prediction) -> int:
        cur = self._conn.execute(
            "INSERT INTO predictions (ts, symbol, timeframe, profile, news_sources, news_score, news_confidence, news_rationale,"
            " technical_score, technical_reasons_json, final_score, final_direction,"
            " final_confidence, price_at_prediction)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (p.ts.isoformat(), p.symbol, p.timeframe, p.profile, p.news_sources, p.news_score, p.news_confidence, p.news_rationale,
             p.technical_score, json.dumps(p.technical_reasons), p.final_score,
             p.final_direction.value, p.final_confidence, p.price_at_prediction),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_watchlist(self, symbol: str, name: str | None = None, sector: str | None = None,
                         notes: str | None = None, sources: str = "google",
                         timeframes: str = "1d", profiles: str = "balanced") -> None:
        self._conn.execute(
            "INSERT INTO watchlist (symbol, name, sector, notes, sources, timeframes, profiles)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(symbol) DO UPDATE SET"
            " name=excluded.name, sector=excluded.sector, notes=excluded.notes,"
            " sources=excluded.sources, timeframes=excluded.timeframes, profiles=excluded.profiles",
            (symbol.upper(), name, sector, notes, sources, timeframes, profiles),
        )
        self._conn.commit()

    def list_watchlist(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute("SELECT * FROM watchlist ORDER BY symbol ASC")
        return cur.fetchall()

    def record_comparison(self, *, ts: str, symbol: str, profile: str, timeframe: str,
                          news_score: float, technical_score: float, final_score: float,
                          final_direction: str, final_confidence: float,
                          news_sources: str, hit: int | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO comparison_runs (ts, symbol, profile, timeframe, news_score, technical_score, final_score, final_direction, final_confidence, news_sources, hit)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, symbol, profile, timeframe, news_score, technical_score, final_score,
             final_direction, final_confidence, news_sources, hit),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def unresolved(self) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT id, ts, symbol, timeframe, final_direction, price_at_prediction"
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

    def summary_by_profile(self, days: int = 30) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT profile, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            " WHERE ts >= datetime('now', ?)"
            " GROUP BY profile"
            " ORDER BY total DESC",
            (f"-{days} days",),
        )
        return cur.fetchall()

    def summary_by_timeframe(self, days: int = 30) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT timeframe, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            " WHERE ts >= datetime('now', ?)"
            " GROUP BY timeframe"
            " ORDER BY total DESC",
            (f"-{days} days",),
        )
        return cur.fetchall()

    def summary_by_direction(self, days: int = 30) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT final_direction, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits"
            " FROM predictions"
            " WHERE ts >= datetime('now', ?)"
            " GROUP BY final_direction"
            " ORDER BY total DESC",
            (f"-{days} days",),
        )
        return cur.fetchall()

    def summary_by_symbol(self, days: int = 30, limit: int = 20) -> list[sqlite3.Row]:
        self._conn.row_factory = sqlite3.Row
        cur = self._conn.execute(
            "SELECT symbol, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            " WHERE ts >= datetime('now', ?)"
            " GROUP BY symbol"
            " ORDER BY total DESC, hits DESC"
            " LIMIT ?",
            (f"-{days} days", limit),
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
