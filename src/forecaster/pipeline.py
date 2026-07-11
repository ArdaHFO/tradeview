"""Orchestration: watchlist -> parallel news/technical -> fusion -> record."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .fusion.combine import combine
from .models import NewsVerdict, Prediction, TechnicalVerdict
from .news.fetch import fetch_articles
from .news.sentiment import analyze_news
from .storage import backfill
from .storage.recorder import PredictionRecorder
from .technical.data import fetch_daily_bars
from .technical.scorer import score_technical

log = logging.getLogger(__name__)


def load_watchlist(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _news_pipeline(item: dict, cfg: Config) -> NewsVerdict:
    articles = fetch_articles(item["symbol"], item.get("name"), cfg)
    return analyze_news(item["symbol"], articles, cfg)


def _technical_pipeline(item: dict, cfg: Config) -> tuple[TechnicalVerdict, float | None]:
    bars = fetch_daily_bars(item["symbol"], cfg)
    verdict = score_technical(item["symbol"], bars)
    price = bars[-1].close if bars else None
    return verdict, price


def run_for_symbols(symbols: list[dict], cfg: Config, progress_cb=None) -> list[Prediction]:
    """Run the news+technical+fusion pipeline for an ad-hoc list of symbols.

    `symbols` is a list of {"symbol": ..., "name": ...} dicts (name optional).
    """
    def report(msg: str) -> None:
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    report("resolving prior predictions...")
    backfill.run(cfg)

    report(f"fetching news + technicals for {len(symbols)} symbol(s)...")
    with ThreadPoolExecutor(max_workers=max(4, len(symbols) * 2)) as ex:
        news_futures = {item["symbol"]: ex.submit(_news_pipeline, item, cfg) for item in symbols}
        tech_futures = {item["symbol"]: ex.submit(_technical_pipeline, item, cfg) for item in symbols}

    predictions: list[Prediction] = []
    recorder = PredictionRecorder(cfg.db_path)
    try:
        for item in symbols:
            symbol = item["symbol"]
            news_verdict = news_futures[symbol].result()
            tech_verdict, price = tech_futures[symbol].result()
            if price is None:
                log.warning("skipping %s: no technical price data", symbol)
                continue
            prediction = combine(news_verdict, tech_verdict, price, cfg)
            recorder.record(prediction)
            predictions.append(prediction)
    finally:
        recorder.close()

    report("done.")
    return predictions


def run_daily(cfg: Config) -> list[Prediction]:
    """CLI entry point: run for the watchlist.json symbols."""
    watchlist = load_watchlist(cfg.watchlist_path)
    return run_for_symbols(watchlist, cfg)
