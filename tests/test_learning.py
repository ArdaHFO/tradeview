import random
from datetime import datetime, timedelta, timezone

from forecaster.config import Config
from forecaster.learning import dataset as ds
from forecaster.learning import train as trainmod
from forecaster.learning.features import (
    FEATURE_NAMES, features_from_bars, features_series, to_vector,
)
from forecaster.learning.metrics import accuracy, auc, brier
from forecaster.learning.model import LogisticRegression
from forecaster.models import Bar


def _bars(trend: float, count: int = 220) -> list[Bar]:
    s = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [
        Bar(ts=s + timedelta(days=i), open=100 + i * trend, high=100 + i * trend + 1,
            low=100 + i * trend - 1, close=100 + i * trend, volume=1_000_000)
        for i in range(count)
    ]


# --- features -------------------------------------------------------------
def test_feature_vector_shape_and_range():
    feats = features_from_bars(_bars(0.3), news_score=0.5)
    assert set(feats) == set(FEATURE_NAMES)
    assert feats["news"] == 0.5
    vec = to_vector(feats)
    assert len(vec) == len(FEATURE_NAMES)
    assert all(-1.0 <= v <= 1.0 for v in vec)


def test_features_series_warmup_is_none_then_populated():
    series = features_series(_bars(0.3))
    assert series[0] is None                      # warm-up
    assert series[-1] is not None
    assert sum(1 for f in series if f is not None) > 100


def test_features_none_when_too_few_bars():
    assert features_from_bars(_bars(0.3, count=10)) is None


# --- model ----------------------------------------------------------------
def test_model_learns_separable_pattern():
    random.seed(1)
    X, y = [], []
    for _ in range(500):
        a, b, c = (random.uniform(-1, 1) for _ in range(3))
        X.append([a, b, c])
        y.append(1 if a + b > 0 else 0)
    m = LogisticRegression(["a", "b", "c"]).fit(X[:400], y[:400])
    p = [m.predict_proba(x) for x in X[400:]]
    assert accuracy(y[400:], p) > 0.85
    assert auc(y[400:], p) > 0.9
    assert abs(m.weights[2]) < abs(m.weights[0])  # irrelevant feature down-weighted


def test_model_json_roundtrip(tmp_path):
    m = LogisticRegression(list(FEATURE_NAMES)).fit([to_vector(features_from_bars(_bars(0.3)))] * 60,
                                                    [1] * 30 + [0] * 30)
    path = tmp_path / "m.json"
    m.save(str(path))
    m2 = LogisticRegression.load(str(path))
    x = to_vector(features_from_bars(_bars(0.3)))
    assert abs(m.predict_proba(x) - m2.predict_proba(x)) < 1e-9


# --- metrics --------------------------------------------------------------
def test_metrics_perfect_and_random():
    y = [1, 0, 1, 0, 1, 0]
    assert accuracy(y, [1, 0, 1, 0, 1, 0]) == 1.0
    assert auc(y, [0.9, 0.1, 0.8, 0.2, 0.7, 0.3]) == 1.0
    assert brier(y, [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]) == 0.0
    assert abs(auc(y, [0.5] * 6) - 0.5) < 1e-9


# --- dataset (backtest bootstrap) -----------------------------------------
def test_build_backtest_dataset_labels(monkeypatch):
    monkeypatch.setattr(ds, "fetch_bars", lambda sym, cfg, timeframe="1d", period="5y": _bars(0.5))
    data = ds.build_backtest_dataset(["UP"], Config(groq_api_key=""))
    assert "UP" in data
    X, y = data["UP"]
    assert len(X) == len(y) and len(X) > 100
    assert len(X[0]) == len(FEATURE_NAMES)
    assert sum(y) / len(y) > 0.9  # steady uptrend -> almost all "up" next day


# --- train / evaluate -----------------------------------------------------
def test_train_beats_baseline_and_saves(monkeypatch, tmp_path):
    # Craft a dataset the trend baseline can't solve but the model can: the
    # label depends on the "news" feature (index 9), while trend (index 0) is
    # random noise.
    random.seed(2)
    ni = FEATURE_NAMES.index("news")
    ti = FEATURE_NAMES.index("trend_sma")
    X, y = [], []
    for _ in range(600):
        row = [0.0] * len(FEATURE_NAMES)
        row[ti] = random.choice([-1.0, 1.0])          # baseline sees only noise
        row[ni] = random.uniform(-1, 1)
        X.append(row)
        y.append(1 if row[ni] > 0 else 0)
    monkeypatch.setattr(trainmod, "build_backtest_dataset", lambda *a, **k: {"SYM": (X, y)})

    path = tmp_path / "model.json"
    model, report = trainmod.train_and_evaluate(["SYM"], Config(groq_api_key=""), save_path=str(path))
    assert report["model"]["accuracy"] > report["baseline_trend"]["accuracy"]
    assert report["beats_baseline"] is True
    assert report["saved"] is True and path.exists()
    assert abs(report["weights"]["news"]) > abs(report["weights"]["trend_sma"])


def test_load_model_missing_returns_none(tmp_path):
    assert trainmod.load_model(str(tmp_path / "nope.json")) is None
