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


def _split(dataset, test_frac: float):
    """Chronological per-symbol split, then concatenate. No look-ahead."""
    Xtr: list[list[float]] = []
    ytr: list[int] = []
    Xte: list[list[float]] = []
    yte: list[int] = []
    for _symbol, (X, y) in dataset.items():
        cut = int(len(X) * (1 - test_frac))
        Xtr += X[:cut]; ytr += y[:cut]
        Xte += X[cut:]; yte += y[cut:]
    return Xtr, ytr, Xte, yte


def train_and_evaluate(symbols: list[str], cfg: Config, timeframe: str = "1d",
                       test_frac: float = 0.3, save_path: str | None = None) -> tuple[LogisticRegression | None, dict]:
    dataset = build_backtest_dataset(symbols, cfg, timeframe)
    Xtr, ytr, Xte, yte = _split(dataset, test_frac)
    if len(Xtr) < 50 or len(Xte) < 20:
        return None, {"error": "not enough data to train/evaluate",
                      "n_train": len(Xtr), "n_test": len(Xte)}

    model = LogisticRegression(FEATURE_NAMES).fit(Xtr, ytr)
    p_model = [model.predict_proba(x) for x in Xte]
    # Baseline: follow the long-term trend (up if SMA50 > SMA200), the kind of
    # rule the hand-tuned scorer leans on. The model has to beat this to ship.
    p_base = [1.0 if x[_TREND_IDX] > 0 else 0.0 for x in Xte]

    report = {
        "symbols": len(dataset),
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
