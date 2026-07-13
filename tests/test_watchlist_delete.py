from forecaster.storage.recorder import PredictionRecorder


def test_delete_watchlist_removes_only_that_symbol(tmp_path):
    db_path = tmp_path / "predictions.db"
    recorder = PredictionRecorder(db_path)
    try:
        recorder.upsert_watchlist("AAPL", "Apple")
        recorder.upsert_watchlist("THYAO.IS", "Turk Hava Yollari")
        assert {row["symbol"] for row in recorder.list_watchlist()} == {"AAPL", "THYAO.IS"}

        recorder.delete_watchlist("AAPL")
        remaining = recorder.list_watchlist()
        assert {row["symbol"] for row in remaining} == {"THYAO.IS"}
    finally:
        recorder.close()


def test_delete_watchlist_scoped_per_user(tmp_path):
    db_path = tmp_path / "predictions.db"
    recorder = PredictionRecorder(db_path)
    try:
        recorder.upsert_watchlist("AAPL", "Apple", user_id=1)
        recorder.upsert_watchlist("AAPL", "Apple", user_id=2)
        recorder.delete_watchlist("AAPL", user_id=1)
        assert recorder.list_watchlist(user_id=1) == []
        assert len(recorder.list_watchlist(user_id=2)) == 1
    finally:
        recorder.close()
