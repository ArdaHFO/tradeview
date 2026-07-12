"""Backtest bootstrap: turn years of price history into labelled examples so
the model can be trained on day one, without waiting for live predictions to
accumulate.

For each symbol we compute the feature vector at every historical day and label
it with the realised direction `horizon` bars later. News is unknown in history,
so its feature stays 0.0 — the bootstrap model is technical/momentum only; the
news feature starts contributing once live predictions (which do have a news
score) are folded in.

Returns a per-symbol dict so callers can split each timeline chronologically
(walk-forward) without leaking the future into the past.
"""
from __future__ import annotations

import logging

from ..config import Config
from ..technical.data import fetch_bars
from .features import features_series, market_index_for, to_vector

log = logging.getLogger(__name__)

Dataset = dict[str, tuple[list[list[float]], list[int]]]


def build_backtest_dataset(symbols: list[str], cfg: Config, timeframe: str = "1d",
                           horizon: int = 1, period: str = "10y") -> Dataset:
    # One index fetch per market, shared by every symbol trading under it.
    market_cache: dict[str, list] = {}

    def market_bars_for(symbol: str):
        index = market_index_for(symbol)
        if index not in market_cache:
            try:
                market_cache[index] = fetch_bars(index, cfg, timeframe, period=period)
            except Exception as exc:  # pragma: no cover - network hiccups
                log.warning("dataset: market index fetch failed for %s: %s", index, exc)
                market_cache[index] = []
        return market_cache[index]

    data: Dataset = {}
    for symbol in symbols:
        try:
            bars = fetch_bars(symbol, cfg, timeframe, period=period)
        except Exception as exc:  # pragma: no cover - network hiccups
            log.warning("dataset: fetch failed for %s: %s", symbol, exc)
            continue
        if len(bars) <= horizon:
            continue
        closes = [b.close for b in bars]
        series = features_series(bars, market_bars_for(symbol))
        rows_x: list[list[float]] = []
        rows_y: list[int] = []
        for i in range(len(bars) - horizon):
            feat = series[i]
            if feat is None:
                continue
            rows_x.append(to_vector(feat))
            rows_y.append(1 if closes[i + horizon] > closes[i] else 0)
        if rows_x:
            data[symbol] = (rows_x, rows_y)
            log.info("dataset: %s -> %d examples", symbol, len(rows_x))
    return data
