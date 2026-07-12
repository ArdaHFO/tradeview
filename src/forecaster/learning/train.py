"""Train the learned model and report honest, out-of-sample numbers.

Walk-forward split: each symbol's timeline is cut chronologically (first part
train, last part test) so we never train on the future. The model is only saved
if it beats a trend-following baseline on the held-out set — otherwise we keep
the hand-tuned fusion and say so.
"""
from __future__ import annotations

import logging

from ..config import Config
from .dataset import build_backtest_dataset
from .features import FEATURE_NAMES
from .metrics import accuracy, auc, brier, calibration
from .model import LogisticRegression

log = logging.getLogger(__name__)

_TREND_IDX = FEATURE_NAMES.index("trend_sma")


def _split(dataset, test_frac: float, embargo: int):
    """Chronological per-symbol split, then concatenate. No look-ahead.

    An `embargo` gap (= the label horizon) is purged between train and test so
    that training labels — which peek `horizon` bars ahead — never depend on
    prices that fall inside the test period (purged walk-forward).
    """
    Xtr: list[list[float]] = []
    ytr: list[int] = []
    Xte: list[list[float]] = []
    yte: list[int] = []
    for _symbol, (X, y) in dataset.items():
        cut = int(len(X) * (1 - test_frac))
        train_end = max(0, cut - embargo)
        Xtr += X[:train_end]; ytr += y[:train_end]
        Xte += X[cut:]; yte += y[cut:]
    return Xtr, ytr, Xte, yte


def train_and_evaluate(symbols: list[str], cfg: Config, timeframe: str = "1d",
                       horizon: int = 60, test_frac: float = 0.3,
                       save_path: str | None = None) -> tuple[LogisticRegression | None, dict]:
    dataset = build_backtest_dataset(symbols, cfg, timeframe, horizon=horizon)
    Xtr, ytr, Xte, yte = _split(dataset, test_frac, embargo=horizon)
    if len(Xtr) < 50 or len(Xte) < 20:
        return None, {"error": "not enough data to train/evaluate",
                      "n_train": len(Xtr), "n_test": len(Xte)}

    model = LogisticRegression(FEATURE_NAMES, l2=0.02, epochs=600).fit(Xtr, ytr)
    p_model = [model.predict_proba(x) for x in Xte]
    # Baseline: follow the long-term trend (up if SMA50 > SMA200), the kind of
    # rule the hand-tuned scorer leans on. The model has to beat this to ship.
    p_base = [1.0 if x[_TREND_IDX] > 0 else 0.0 for x in Xte]

    report = {
        "symbols": len(dataset),
        "horizon": horizon,
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "positive_rate_test": round(sum(yte) / len(yte), 3),
        "model": {
            "accuracy": round(accuracy(yte, p_model), 4),
            "auc": round(auc(yte, p_model), 4),
            "brier": round(brier(yte, p_model), 4),
        },
        "baseline_trend": {"accuracy": round(accuracy(yte, p_base), 4)},
        "weights": {name: round(w, 4) for name, w in zip(FEATURE_NAMES, model.weights)},
        "calibration": calibration(yte, p_model, bins=10),
    }
    beats_baseline = (report["model"]["accuracy"] >= report["baseline_trend"]["accuracy"]
                      and report["model"]["auc"] > 0.5)
    report["beats_baseline"] = beats_baseline
    report["saved"] = False
    # Embed a compact, honest summary in the model file so the UI can show it.
    from datetime import datetime, timezone
    model.meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon,
        "symbols": report["symbols"],
        "n_train": report["n_train"],
        "n_test": report["n_test"],
        "accuracy": report["model"]["accuracy"],
        "auc": report["model"]["auc"],
        "brier": report["model"]["brier"],
        "baseline_accuracy": report["baseline_trend"]["accuracy"],
        "weights": report["weights"],
    }
    if save_path and beats_baseline:
        model.save(save_path)
        report["saved"] = True
        report["saved_path"] = save_path
        log.info("learned model saved to %s (accuracy %.3f, auc %.3f)",
                 save_path, report["model"]["accuracy"], report["model"]["auc"])
    else:
        log.info("learned model NOT saved (beats_baseline=%s)", beats_baseline)
    return model, report


def load_model(path: str) -> LogisticRegression | None:
    try:
        return LogisticRegression.load(path)
    except (OSError, ValueError, KeyError):
        return None
