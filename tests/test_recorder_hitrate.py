import sqlite3
from datetime import datetime, timezone

from forecaster.models import Direction, Prediction
from forecaster.storage.recorder import PredictionRecorder


def _prediction(symbol="AAPL", score=0.1, timeframe="1d", profile="balanced"):
    return Prediction(
        ts=datetime.now(timezone.utc), symbol=symbol, news_score=score, news_confidence=0.1,
        news_rationale="x", technical_score=0.2, technical_reasons=[], final_score=score,
        final_direction=Direction.UP, final_confidence=0.5, price_at_prediction=100.0,
        timeframe=timeframe, profile=profile,
    )


def test_hit_rate_counts_resolved_predictions(tmp_path):
    db_path = str(tmp_path / "hitrate.db")
    rec = PredictionRecorder(db_path)
    pid = rec.record(_prediction(), user_id=1)
    rec.resolve(pid, 101.0, Direction.UP, True)
    hits, total = rec.hit_rate(7, user_id=1)
    assert (hits, total) == (1, 1)
    rec.close()


def test_same_day_pending_prediction_is_updated_not_duplicated(tmp_path):
    db_path = str(tmp_path / "dedupe.db")
    rec = PredictionRecorder(db_path)
    id1 = rec.record(_prediction(score=0.1), user_id=1)
    id2 = rec.record(_prediction(score=0.2), user_id=1)
    assert id1 == id2
    rows = rec.recent(limit=10, user_id=1)
    assert len(rows) == 1
    assert rows[0]["final_score"] == 0.2
    rec.close()


def test_resolved_same_day_prediction_gets_a_new_row(tmp_path):
    db_path = str(tmp_path / "dedupe2.db")
    rec = PredictionRecorder(db_path)
    id1 = rec.record(_prediction(score=0.1), user_id=1)
    rec.resolve(id1, 101.0, Direction.UP, True)
    id2 = rec.record(_prediction(score=0.3), user_id=1)
    assert id2 != id1
    assert len(rec.recent(limit=10, user_id=1)) == 2
    rec.close()


def test_different_timeframe_same_day_is_not_deduped(tmp_path):
    db_path = str(tmp_path / "dedupe3.db")
    rec = PredictionRecorder(db_path)
    id1 = rec.record(_prediction(timeframe="1d"), user_id=1)
    id2 = rec.record(_prediction(timeframe="1h"), user_id=1)
    assert id1 != id2
    assert len(rec.recent(limit=10, user_id=1)) == 2
    rec.close()


def test_hitrate_v2_migration_resets_stale_graded_rows(tmp_path):
    db_path = str(tmp_path / "migration.db")
    # Seed a pre-migration-style row directly, bypassing the recorder so no
    # migration has run against this file yet.
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER,
            ts TEXT NOT NULL, symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL DEFAULT '1d', profile TEXT NOT NULL DEFAULT 'balanced',
            news_sources TEXT NOT NULL DEFAULT 'google',
            news_score REAL NOT NULL, news_confidence REAL NOT NULL, news_rationale TEXT NOT NULL,
            technical_score REAL NOT NULL, technical_reasons_json TEXT NOT NULL,
            final_score REAL NOT NULL, final_direction TEXT NOT NULL, final_confidence REAL NOT NULL,
            price_at_prediction REAL NOT NULL, actual_next_close REAL, actual_direction TEXT, hit INTEGER
        );
    """)
    conn.execute(
        "INSERT INTO predictions (user_id, ts, symbol, news_score, news_confidence, news_rationale,"
        " technical_score, technical_reasons_json, final_score, final_direction, final_confidence,"
        " price_at_prediction, actual_next_close, actual_direction, hit)"
        " VALUES (1,'2026-07-01T00:00:00+00:00','AAPL',0.1,0.1,'x',0.1,'[]',0.1,'UP',0.5,100.0,100.01,'NEUTRAL',0)"
    )
    conn.commit()
    conn.close()

    rec = PredictionRecorder(db_path)
    row = rec.recent(limit=10, user_id=1)[0]
    assert row["hit"] is None
    assert row["actual_direction"] is None
    rec.close()

    # Re-opening must be a no-op (idempotent) and must not wipe fresh grades.
    rec2 = PredictionRecorder(db_path)
    pid = rec2.record(_prediction(), user_id=1)
    rec2.resolve(pid, 101.0, Direction.UP, True)
    rec2.close()

    rec3 = PredictionRecorder(db_path)
    hits, total = rec3.hit_rate(3650, user_id=1)
    assert (hits, total) == (1, 1)
    rec3.close()
