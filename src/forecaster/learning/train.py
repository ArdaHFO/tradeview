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


def _split(dataset, test_frac: float, embargo: int, val_frac: float = 0.0):
    """Chronological per-symbol split, then concatenate. No look-ahead.

    An `embargo` gap (= the label horizon) is purged between the segments so
    that training labels — which peek `horizon` bars ahead — never depend on
    prices inside the later segment (purged walk-forward). With `val_frac`,
    carve a validation slice out of the *end* of the train segment for
    hyper-parameter selection.
    """
    Xtr: list[list[float]] = []
    ytr: list[int] = []
    Xval: list[list[float]] = []
    yval: list[int] = []
    Xte: list[list[float]] = []
    yte: list[int] = []
    for _symbol, (X, y) in dataset.items():
        cut = int(len(X) * (1 - test_frac))
        train_end = max(0, cut - embargo)
        if val_frac > 0:
            val_cut = int(train_end * (1 - val_frac))
            fit_end = max(0, val_cut - embargo)
            Xtr += X[:fit_end]; ytr += y[:fit_end]
            Xval += X[val_cut:train_end]; yval += y[val_cut:train_end]
        else:
            Xtr += X[:train_end]; ytr += y[:train_end]
        Xte += X[cut:]; yte += y[cut:]
    return Xtr, ytr, Xval, yval, Xte, yte


def _selective_bands(yte: list[int], p_model: list[float]) -> list[dict]:
    """Accuracy when acting only on higher-conviction signals (|P-0.5| >= t).

    This is how the model is meant to be used professionally: fewer, better
    signals. Coverage tells you how often it speaks at that bar."""
    bands = []
    for t in (0.0, 0.05, 0.10, 0.15, 0.20):
        idx = [i for i, p in enumerate(p_model) if abs(p - 0.5) >= t]
        if not idx:
            continue
        sel_y = [yte[i] for i in idx]
        sel_p = [p_model[i] for i in idx]
        bands.append({
            "min_conviction": t,
            "coverage": round(len(idx) / len(p_model), 3),
            "accuracy": round(accuracy(sel_y, sel_p), 4),
            "n": len(idx),
        })
    return bands


def train_and_evaluate(symbols: list[str], cfg: Config, timeframe: str = "1d",
                       horizon: int = 60, test_frac: float = 0.3,
                       save_path: str | None = None) -> tuple[LogisticRegression | None, dict]:
    dataset = build_backtest_dataset(symbols, cfg, timeframe, horizon=horizon)
    Xtr, ytr, Xval, yval, Xte, yte = _split(dataset, test_frac, embargo=horizon, val_frac=0.15)
    if len(Xtr) < 50 or len(Xte) < 20:
        return None, {"error": "not enough data to train/evaluate",
                      "n_train": len(Xtr), "n_test": len(Xte)}

    # Small, honest hyper-parameter search: pick L2 by validation AUC (a slice
    # of the train timeline — the test period stays untouched).
    best_l2, best_val_auc = 0.02, -1.0
    if len(Xval) >= 50:
        for l2 in (0.005, 0.02, 0.08):
            cand = LogisticRegression(FEATURE_NAMES, l2=l2, epochs=600).fit(Xtr, ytr)
            val_auc = auc(yval, [cand.predict_proba(x) for x in Xval])
            if val_auc > best_val_auc:
                best_l2, best_val_auc = l2, val_auc

    # Refit on the full train window (train + validation) with the chosen L2.
    Xfit, yfit = Xtr + Xval, ytr + yval
    model = LogisticRegression(FEATURE_NAMES, l2=best_l2, epochs=600).fit(Xfit, yfit)
    p_model = [model.predict_proba(x) for x in Xte]
    # Baselines the model has to justify itself against: the long-term trend
    # rule (what the hand-tuned scorer leans on) and always-up (the base rate).
    p_base = [1.0 if x[_TREND_IDX] > 0 else 0.0 for x in Xte]
    p_up = [1.0] * len(yte)

    report = {
        "symbols": len(dataset),
        "horizon": horizon,
        "n_train": len(Xfit),
        "n_test": len(Xte),
        "l2": best_l2,
        "positive_rate_test": round(sum(yte) / len(yte), 3),
        "model": {
            "accuracy": round(accuracy(yte, p_model), 4),
            "auc": round(auc(yte, p_model), 4),
            "brier": round(brier(yte, p_model), 4),
        },
        "baseline_trend": {"accuracy": round(accuracy(yte, p_base), 4)},
        "baseline_always_up": {"accuracy": round(accuracy(yte, p_up), 4)},
        "selective": _selective_bands(yte, p_model),
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
        "base_rate": report["baseline_always_up"]["accuracy"],
        "selective": report["selective"],
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
