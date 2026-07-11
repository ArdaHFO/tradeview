from forecaster.storage.recorder import PredictionRecorder


def test_app_settings_roundtrip(tmp_path):
    db_path = tmp_path / "predictions.db"
    recorder = PredictionRecorder(db_path)
    try:
        defaults = recorder.get_settings()
        assert defaults["groq_model"] == "llama-3.3-70b-versatile"
        assert defaults["news_weight"] == "0.5"

        recorder.upsert_settings({"groq_model": "llama-3.1-8b-instant", "news_weight": "0.7"})
        updated = recorder.get_settings()
        assert updated["groq_model"] == "llama-3.1-8b-instant"
        assert updated["news_weight"] == "0.7"
    finally:
        recorder.close()