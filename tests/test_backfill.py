from datetime import datetime, timedelta, timezone

from forecaster import models
from forecaster.config import Config
from forecaster.models import Bar, Direction
from forecaster.storage import backfill
from forecaster.storage.recorder import PredictionRecorder


def _bar(ts, close):
    return Bar(ts=ts, open=close, high=close + 1, low=close - 1, close=close, volume=1.0)


def test_actual_direction_scales_threshold_by_timeframe():
    # A 0.2% move is NEUTRAL for a 1d prediction (threshold 0.30%) but a
    # real UP for a 30m prediction (threshold 0.10%).
    assert backfill._actual_direction(100.0, 100.2, "1d") == Direction.NEUTRAL
    assert backfill._actual_direction(100.0, 100.2, "30m") == Direction.UP


def test_first_bar_after_returns_none_when_no_new_bar():
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    bars = [_bar(cutoff - timedelta(hours=1), 100.0)]
    assert backfill._first_bar_after(bars, cutoff) is None


def test_first_bar_after_returns_first_newer_bar():
    cutoff = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    bars = [
        _bar(cutoff - timedelta(hours=1), 100.0),
        _bar(cutoff + timedelta(hours=1), 103.0),
        _bar(cutoff + timedelta(hours=2), 105.0),
    ]
    found = backfill._first_bar_after(bars, cutoff)
    assert found is not None and found.close == 103.0


def test_run_leaves_prediction_pending_until_new_bar_closes(monkeypatch, tmp_path):
    db_path = str(tmp_path / "backfill_test.db")
    rec = PredictionRecorder(db_path)
    pred_ts = datetime.now(timezone.utc) - timedelta(hours=2)
    p = models.Prediction(
        ts=pred_ts, symbol="AAPL", news_score=0.1, news_confidence=0.1,
        news_rationale="x", technical_score=0.2, technical_reasons=[], final_score=0.2,
        final_direction=Direction.UP, final_confidence=0.5, price_at_prediction=100.0,
        timeframe="1d",
    )
    pid = rec.record(p, user_id=None)
    rec.close()

    cfg = Config(groq_api_key="", db_path=db_path)

    # No new bar yet -> stays unresolved.
    monkeypatch.setattr(backfill, "fetch_bars",
                         lambda symbol, cfg, timeframe: [_bar(pred_ts - timedelta(hours=1), 100.0)])
    backfill.run(cfg)
    rec2 = PredictionRecorder(db_path)
    assert len(rec2.unresolved(user_id=None)) == 1
    rec2.close()

    # A bar closed after the prediction -> resolves.
    monkeypatch.setattr(backfill, "fetch_bars",
                         lambda symbol, cfg, timeframe: [_bar(pred_ts + timedelta(hours=1), 103.0)])
    backfill.run(cfg)
    rec3 = PredictionRecorder(db_path)
    assert len(rec3.unresolved(user_id=None)) == 0
    row = rec3.recent(limit=1, user_id=None)[0]
    assert row["hit"] == 1
    rec3.close()
