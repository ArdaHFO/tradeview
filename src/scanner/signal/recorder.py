"""Persist every emitted signal to SQLite for later performance analysis."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ..models import Signal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL,
    side TEXT NOT NULL,
    entry REAL NOT NULL,
    stop REAL NOT NULL,
    target REAL NOT NULL,
    confluence REAL NOT NULL,
    scores_json TEXT NOT NULL,
    position_size INTEGER NOT NULL,
    risk_dollars REAL NOT NULL,
    reasons_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts ON signals(symbol, ts);
"""


class SignalRecorder:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, sig: Signal) -> int:
        cur = self._conn.execute(
            "INSERT INTO signals (ts, symbol, setup, side, entry, stop, target,"
            " confluence, scores_json, position_size, risk_dollars, reasons_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sig.ts.isoformat(), sig.symbol, sig.setup.value, sig.side.value,
             sig.entry, sig.stop, sig.target, sig.confluence,
             json.dumps(sig.scores.as_dict()), sig.position_size,
             sig.risk_dollars, json.dumps(sig.reasons)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM signals").fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()
