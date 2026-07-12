"""A small, dependency-free logistic regression.

Deliberately pure-Python (no numpy/sklearn): a handful of features and a few
hundred gradient-descent epochs are plenty fast, and it keeps the model
interpretable — the learned weights say which signals actually mattered.

predict_proba(x) returns P(price goes up); the fusion layer maps that to a
score in [-1, 1].
"""
from __future__ import annotations

import json
import math


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class LogisticRegression:
    def __init__(self, feature_names: list[str] | tuple[str, ...],
                 l2: float = 0.01, lr: float = 0.2, epochs: int = 500) -> None:
        self.feature_names = list(feature_names)
        self.l2 = l2
        self.lr = lr
        self.epochs = epochs
        m = len(self.feature_names)
        self.weights = [0.0] * m
        self.bias = 0.0
        # Feature standardisation (fit on training data) — logistic GD is much
        # better behaved when inputs are on a common scale.
        self.mean = [0.0] * m
        self.std = [1.0] * m
        # Training report (accuracy/auc/…) so the UI can show honest numbers.
        self.meta: dict = {}

    # -- standardisation --------------------------------------------------
    def _fit_scaler(self, X: list[list[float]]) -> None:
        n = len(X)
        for j in range(len(self.feature_names)):
            col = [row[j] for row in X]
            mu = sum(col) / n
            var = sum((c - mu) ** 2 for c in col) / n
            self.mean[j] = mu
            self.std[j] = math.sqrt(var) or 1.0

    def _z(self, x: list[float]) -> list[float]:
        return [(x[j] - self.mean[j]) / self.std[j] for j in range(len(x))]

    # -- training ---------------------------------------------------------
    def fit(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        if not X:
            raise ValueError("cannot fit on an empty dataset")
        self._fit_scaler(X)
        try:
            import numpy as np  # optional: vectorised GD for large datasets
        except ImportError:
            return self._fit_python(X, y)

        Xs = np.array([self._z(x) for x in X], dtype=float)
        yv = np.array(y, dtype=float)
        n, m = Xs.shape
        w = np.zeros(m)
        b = 0.0
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-np.clip(Xs.dot(w) + b, -30.0, 30.0)))
            err = p - yv
            w -= self.lr * (Xs.T.dot(err) / n + self.l2 * w)
            b -= self.lr * (err.sum() / n)
        self.weights = w.tolist()
        self.bias = float(b)
        return self

    def _fit_python(self, X: list[list[float]], y: list[int]) -> "LogisticRegression":
        Xs = [self._z(x) for x in X]
        n = len(Xs)
        m = len(self.feature_names)
        for _ in range(self.epochs):
            grad_w = [0.0] * m
            grad_b = 0.0
            for xi, yi in zip(Xs, y):
                p = _sigmoid(self.bias + sum(self.weights[j] * xi[j] for j in range(m)))
                err = p - yi
                for j in range(m):
                    grad_w[j] += err * xi[j]
                grad_b += err
            for j in range(m):
                self.weights[j] -= self.lr * (grad_w[j] / n + self.l2 * self.weights[j])
            self.bias -= self.lr * (grad_b / n)
        return self

    # -- inference --------------------------------------------------------
    def predict_proba(self, x: list[float]) -> float:
        xs = self._z(x)
        return _sigmoid(self.bias + sum(self.weights[j] * xs[j] for j in range(len(xs))))

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "weights": self.weights,
            "bias": self.bias,
            "mean": self.mean,
            "std": self.std,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LogisticRegression":
        model = cls(d["feature_names"])
        model.weights = list(d["weights"])
        model.bias = float(d["bias"])
        model.mean = list(d["mean"])
        model.std = list(d["std"])
        model.meta = dict(d.get("meta", {}))
        return model

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
