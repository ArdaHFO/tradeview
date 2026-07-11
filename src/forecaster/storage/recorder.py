"""Persist every prediction to SQLite for later accuracy tracking."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import secrets
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any

try:
    import psycopg  # type: ignore[import-not-found]
    from psycopg.rows import dict_row  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for Supabase/Postgres deployments
    psycopg = None
    dict_row = None

from ..models import Direction, Prediction

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
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
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(user_id, symbol, ts);

CREATE TABLE IF NOT EXISTS user_watchlist (
    user_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    notes TEXT,
    sources TEXT DEFAULT 'google',
    timeframes TEXT DEFAULT '1d',
    profiles TEXT DEFAULT 'balanced',
    PRIMARY KEY (user_id, symbol)
);

CREATE TABLE IF NOT EXISTS comparison_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
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

CREATE TABLE IF NOT EXISTS user_app_settings (
    user_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL
    ,PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_ts TEXT NOT NULL,
    expires_ts TEXT NOT NULL
);
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    profile TEXT NOT NULL,
    news_sources TEXT NOT NULL,
    news_score DOUBLE PRECISION NOT NULL,
    news_confidence DOUBLE PRECISION NOT NULL,
    news_rationale TEXT NOT NULL,
    technical_score DOUBLE PRECISION NOT NULL,
    technical_reasons_json TEXT NOT NULL,
    final_score DOUBLE PRECISION NOT NULL,
    final_direction TEXT NOT NULL,
    final_confidence DOUBLE PRECISION NOT NULL,
    price_at_prediction DOUBLE PRECISION NOT NULL,
    actual_next_close DOUBLE PRECISION,
    actual_direction TEXT,
    hit INTEGER
);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(user_id, symbol, ts);

CREATE TABLE IF NOT EXISTS user_watchlist (
    user_id BIGINT NOT NULL,
    symbol TEXT NOT NULL,
    name TEXT,
    sector TEXT,
    notes TEXT,
    sources TEXT DEFAULT 'google',
    timeframes TEXT DEFAULT '1d',
    profiles TEXT DEFAULT 'balanced',
    PRIMARY KEY (user_id, symbol)
);

CREATE TABLE IF NOT EXISTS comparison_runs (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    profile TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    news_score DOUBLE PRECISION NOT NULL,
    technical_score DOUBLE PRECISION NOT NULL,
    final_score DOUBLE PRECISION NOT NULL,
    final_direction TEXT NOT NULL,
    final_confidence DOUBLE PRECISION NOT NULL,
    news_sources TEXT NOT NULL,
    hit INTEGER
);

CREATE TABLE IF NOT EXISTS user_app_settings (
    user_id BIGINT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id BIGINT NOT NULL,
    created_ts TEXT NOT NULL,
    expires_ts TEXT NOT NULL
);
"""

_DEFAULT_SETTINGS: dict[str, str] = {
    "news_weight": "0.5",
    "technical_weight": "0.5",
    "neutral_band": "0.15",
    "groq_model": "llama-3.3-70b-versatile",
    "news_lookback_hours": "24",
    "max_articles_per_symbol": "10",
    "max_symbols_per_run": "10",
    "intraday_lookback_period": "60d",
    "technical_lookback_period": "6mo",
}


class PredictionRecorder:
    def __init__(self, db_path: str | Path) -> None:
        self._dsn = str(db_path)
        self._backend = "postgres" if self._dsn.startswith(("postgres://", "postgresql://")) else "sqlite"
        if self._backend == "postgres":
            if psycopg is None or dict_row is None:
                raise RuntimeError("psycopg is required when DATABASE_URL points to Postgres/Supabase")
            self._conn = psycopg.connect(self._dsn, row_factory=dict_row)
            self._ensure_schema_postgres()
        else:
            self._conn = sqlite3.connect(self._dsn)
            self._conn.executescript(_SQLITE_SCHEMA)
            self._conn.commit()
            self._ensure_user_columns()

    def _ensure_schema_postgres(self) -> None:
        for statement in (part.strip() for part in _POSTGRES_SCHEMA.split(";") if part.strip()):
            self._conn.execute(statement)
        self._conn.commit()

    def _execute(self, query: str, params: tuple[Any, ...] = ()):
        if self._backend == "postgres":
            query = query.replace("?", "%s")
        return self._conn.execute(query, params)

    def _use_dict_rows(self) -> None:
        if self._backend == "sqlite":
            self._conn.row_factory = sqlite3.Row

    def _ensure_user_columns(self) -> None:
        if self._backend != "sqlite":
            return
        self._add_column_if_missing("predictions", "user_id", "INTEGER")
        self._add_column_if_missing("comparison_runs", "user_id", "INTEGER")

    def _table_columns(self, table: str) -> set[str]:
        if self._backend != "sqlite":
            return set()
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return {str(row[1]) for row in cur.fetchall()}

    def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        if self._backend != "sqlite":
            return
        if column not in self._table_columns(table):
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self._conn.commit()

    def _default_user_id(self) -> int:
        user_id = self.get_user_id("default")
        if user_id is None:
            user_id = self.create_user("default", "default")
        return user_id

    def hash_password(self, password: str, salt: str | None = None) -> tuple[str, str]:
        salt_value = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_value.encode("utf-8"), 120_000)
        return digest.hex(), salt_value

    def verify_password(self, password: str, password_hash: str, password_salt: str) -> bool:
        digest, _ = self.hash_password(password, password_salt)
        return secrets.compare_digest(digest, password_hash)

    def create_user(self, username: str, password: str) -> int:
        password_hash, password_salt = self.hash_password(password)
        if self._backend == "postgres":
            cur = self._execute(
                "INSERT INTO users (username, password_hash, password_salt, created_ts) VALUES (?, ?, ?, ?) RETURNING id",
                (username.strip().lower(), password_hash, password_salt, datetime.now(timezone.utc).isoformat()),
            )
            row = cur.fetchone()
            self._conn.commit()
            return int(row["id"])
        cur = self._execute(
            "INSERT INTO users (username, password_hash, password_salt, created_ts) VALUES (?, ?, ?, ?)",
            (username.strip().lower(), password_hash, password_salt, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_user(self, username: str) -> Any | None:
        self._use_dict_rows()
        cur = self._execute("SELECT * FROM users WHERE username = ?", (username.strip().lower(),))
        return cur.fetchone()

    def get_user_id(self, username: str) -> int | None:
        row = self.get_user(username)
        return int(row["id"]) if row else None

    def get_user_by_id(self, user_id: int) -> Any | None:
        self._use_dict_rows()
        cur = self._execute("SELECT * FROM users WHERE id = ?", (user_id,))
        return cur.fetchone()

    def authenticate_user(self, username: str, password: str) -> int | None:
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, str(user["password_hash"]), str(user["password_salt"])):
            return None
        return int(user["id"])

    def create_session(self, user_id: int, ttl_hours: int = 24 * 7) -> str:
        token = secrets.token_urlsafe(32)
        created = datetime.now(timezone.utc)
        expires = created + timedelta(hours=ttl_hours)
        self._execute(
            "INSERT INTO sessions (token, user_id, created_ts, expires_ts) VALUES (?, ?, ?, ?)",
            (token, user_id, created.isoformat(), expires.isoformat()),
        )
        self._conn.commit()
        return token

    def get_session_user_id(self, token: str) -> int | None:
        self._use_dict_rows()
        cur = self._execute(
            "SELECT user_id, expires_ts FROM sessions WHERE token = ?",
            (token,),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            expires = datetime.fromisoformat(str(row["expires_ts"]))
        except ValueError:
            return None
        if expires < datetime.now(timezone.utc):
            self.delete_session(token)
            return None
        return int(row["user_id"])

    def delete_session(self, token: str) -> None:
        self._execute("DELETE FROM sessions WHERE token = ?", (token,))
        self._conn.commit()

    def _scope_user_id(self, user_id: int | None) -> int:
        return user_id if user_id is not None else self._default_user_id()

    def get_settings(self, user_id: int | None = None) -> dict[str, str]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        cur = self._execute("SELECT key, value FROM user_app_settings WHERE user_id = ?", (scoped_user_id,))
        settings = dict(_DEFAULT_SETTINGS)
        for row in cur.fetchall():
            settings[str(row["key"])] = str(row["value"])
        return settings

    def upsert_settings(self, settings: dict[str, str], user_id: int | None = None) -> None:
        scoped_user_id = self._scope_user_id(user_id)
        for key, value in settings.items():
            self._execute(
                "INSERT INTO user_app_settings (user_id, key, value) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value",
                (scoped_user_id, str(key), str(value)),
            )
        self._conn.commit()

    def record(self, p: Prediction, user_id: int | None = None) -> int:
        scoped_user_id = self._scope_user_id(user_id)
        if self._backend == "postgres":
            cur = self._execute(
                "INSERT INTO predictions (user_id, ts, symbol, timeframe, profile, news_sources, news_score, news_confidence, news_rationale,"
                " technical_score, technical_reasons_json, final_score, final_direction,"
                " final_confidence, price_at_prediction)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (scoped_user_id, p.ts.isoformat(), p.symbol, p.timeframe, p.profile, p.news_sources, p.news_score, p.news_confidence, p.news_rationale,
                 p.technical_score, json.dumps(p.technical_reasons), p.final_score,
                 p.final_direction.value, p.final_confidence, p.price_at_prediction),
            )
            row = cur.fetchone()
            self._conn.commit()
            return int(row["id"])
        cur = self._execute(
            "INSERT INTO predictions (user_id, ts, symbol, timeframe, profile, news_sources, news_score, news_confidence, news_rationale,"
            " technical_score, technical_reasons_json, final_score, final_direction,"
            " final_confidence, price_at_prediction)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scoped_user_id, p.ts.isoformat(), p.symbol, p.timeframe, p.profile, p.news_sources, p.news_score, p.news_confidence, p.news_rationale,
             p.technical_score, json.dumps(p.technical_reasons), p.final_score,
             p.final_direction.value, p.final_confidence, p.price_at_prediction),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def upsert_watchlist(self, symbol: str, name: str | None = None, sector: str | None = None,
                         notes: str | None = None, sources: str = "google",
                         timeframes: str = "1d", profiles: str = "balanced",
                         user_id: int | None = None) -> None:
        scoped_user_id = self._scope_user_id(user_id)
        self._execute(
            "INSERT INTO user_watchlist (user_id, symbol, name, sector, notes, sources, timeframes, profiles)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(user_id, symbol) DO UPDATE SET"
            " name=excluded.name, sector=excluded.sector, notes=excluded.notes,"
            " sources=excluded.sources, timeframes=excluded.timeframes, profiles=excluded.profiles",
            (scoped_user_id, symbol.upper(), name, sector, notes, sources, timeframes, profiles),
        )
        self._conn.commit()

    def list_watchlist(self, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        cur = self._execute("SELECT * FROM user_watchlist WHERE user_id = ? ORDER BY symbol ASC", (scoped_user_id,))
        return cur.fetchall()

    def record_comparison(self, *, ts: str, symbol: str, profile: str, timeframe: str,
                          news_score: float, technical_score: float, final_score: float,
                          final_direction: str, final_confidence: float,
                          news_sources: str, hit: int | None = None,
                          user_id: int | None = None) -> int:
        scoped_user_id = self._scope_user_id(user_id)
        if self._backend == "postgres":
            cur = self._execute(
                "INSERT INTO comparison_runs (user_id, ts, symbol, profile, timeframe, news_score, technical_score, final_score, final_direction, final_confidence, news_sources, hit)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
                (scoped_user_id, ts, symbol, profile, timeframe, news_score, technical_score, final_score,
                 final_direction, final_confidence, news_sources, hit),
            )
            row = cur.fetchone()
            self._conn.commit()
            return int(row["id"])
        cur = self._execute(
            "INSERT INTO comparison_runs (user_id, ts, symbol, profile, timeframe, news_score, technical_score, final_score, final_direction, final_confidence, news_sources, hit)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (scoped_user_id, ts, symbol, profile, timeframe, news_score, technical_score, final_score,
             final_direction, final_confidence, news_sources, hit),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def unresolved(self, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        cur = self._execute(
            "SELECT id, ts, symbol, timeframe, final_direction, price_at_prediction"
            " FROM predictions WHERE hit IS NULL AND user_id = ?",
            (scoped_user_id,),
        )
        return cur.fetchall()

    def resolve(self, prediction_id: int, actual_next_close: float,
                actual_direction: Direction, hit: bool) -> None:
        self._execute(
            "UPDATE predictions SET actual_next_close = ?, actual_direction = ?, hit = ?"
            " WHERE id = ?",
            (actual_next_close, actual_direction.value, int(hit), prediction_id),
        )
        self._conn.commit()

    def recent(self, limit: int = 50, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        cur = self._execute(
            "SELECT ts, symbol, final_score, final_direction, final_confidence,"
            " actual_direction, hit FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (scoped_user_id, limit),
        )
        return cur.fetchall()

    def _since_clause(self, days: int) -> tuple[str, tuple[Any, ...]]:
        if self._backend == "postgres":
            return "ts >= NOW() - (%s || ' days')::interval", (days,)
        return "ts >= datetime('now', ?)", (f"-{days} days",)

    def summary_by_profile(self, days: int = 30, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        since_clause, since_params = self._since_clause(days)
        cur = self._execute(
            "SELECT profile, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            f" WHERE user_id = ? AND {since_clause}"
            " GROUP BY profile"
            " ORDER BY total DESC",
            (scoped_user_id, *since_params),
        )
        return cur.fetchall()

    def summary_by_timeframe(self, days: int = 30, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        since_clause, since_params = self._since_clause(days)
        cur = self._execute(
            "SELECT timeframe, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            f" WHERE user_id = ? AND {since_clause}"
            " GROUP BY timeframe"
            " ORDER BY total DESC",
            (scoped_user_id, *since_params),
        )
        return cur.fetchall()

    def summary_by_direction(self, days: int = 30, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        since_clause, since_params = self._since_clause(days)
        cur = self._execute(
            "SELECT final_direction, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits"
            " FROM predictions"
            f" WHERE user_id = ? AND {since_clause}"
            " GROUP BY final_direction"
            " ORDER BY total DESC",
            (scoped_user_id, *since_params),
        )
        return cur.fetchall()

    def summary_by_symbol(self, days: int = 30, limit: int = 20, user_id: int | None = None) -> list[sqlite3.Row]:
        scoped_user_id = self._scope_user_id(user_id)
        self._use_dict_rows()
        since_clause, since_params = self._since_clause(days)
        cur = self._execute(
            "SELECT symbol, COUNT(*) AS total, COALESCE(SUM(hit), 0) AS hits,"
            " ROUND(AVG(final_confidence), 3) AS avg_confidence,"
            " ROUND(AVG(final_score), 3) AS avg_score"
            " FROM predictions"
            f" WHERE user_id = ? AND {since_clause}"
            " GROUP BY symbol"
            " ORDER BY total DESC, hits DESC"
            " LIMIT ?",
            (scoped_user_id, *since_params, limit),
        )
        return cur.fetchall()

    def hit_rate(self, days: int, user_id: int | None = None) -> tuple[int, int]:
        scoped_user_id = self._scope_user_id(user_id)
        since_clause, since_params = self._since_clause(days)
        cur = self._execute(
            "SELECT COUNT(*), COALESCE(SUM(hit), 0) FROM predictions"
            f" WHERE user_id = ? AND hit IS NOT NULL AND {since_clause}",
            (scoped_user_id, *since_params),
        )
        total, hits = cur.fetchone()
        return int(hits), int(total)

    def close(self) -> None:
        self._conn.close()
