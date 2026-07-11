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
from .technical.data import fetch_bars
from .technical.scorer import score_technical

log = logging.getLogger(__name__)


def load_watchlist(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_watchlist_items(cfg: Config) -> list[dict]:
    try:
        with open(cfg.watchlist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _news_pipeline(item: dict, cfg: Config) -> NewsVerdict:
    articles = fetch_articles(item["symbol"], item.get("name"), cfg, item.get("news_sources"))
    return analyze_news(item["symbol"], articles, cfg)


def _technical_pipeline(item: dict, cfg: Config) -> tuple[TechnicalVerdict, float | None]:
    timeframe = str(item.get("timeframe", "1d"))
    bars = fetch_bars(item["symbol"], cfg, timeframe)
    verdict = score_technical(item["symbol"], bars)
    price = bars[-1].close if bars else None
    return verdict, price


def _analysis_meta(item: dict) -> tuple[str, str, str]:
    return (
        str(item.get("timeframe", "1d")),
        str(item.get("profile", "balanced")),
        ",".join(item.get("news_sources") or ["google"]),
    )


def _run_key(item: dict) -> str:
    timeframe = str(item.get("timeframe", "1d"))
    profile = str(item.get("profile", "balanced"))
    news_sources = ",".join(item.get("news_sources") or ["google"])
    return f"{item['symbol']}|{timeframe}|{profile}|{news_sources}"


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
        news_futures = {_run_key(item): ex.submit(_news_pipeline, item, cfg) for item in symbols}
        tech_futures = {_run_key(item): ex.submit(_technical_pipeline, item, cfg) for item in symbols}

    predictions: list[Prediction] = []
    recorder = PredictionRecorder(cfg.db_path)
    try:
        for item in symbols:
            key = _run_key(item)
            symbol = item["symbol"]
            news_verdict = news_futures[key].result()
            tech_verdict, price = tech_futures[key].result()
            if price is None:
                log.warning("skipping %s: no technical price data", symbol)
                continue
            timeframe, profile, news_sources = _analysis_meta(item)
            prediction = combine(
                news_verdict,
                tech_verdict,
                price,
                cfg,
                timeframe=timeframe,
                profile=profile,
                news_sources=news_sources,
            )
            recorder.record(prediction)
            predictions.append(prediction)
    finally:
        recorder.close()

    report("done.")
    return predictions


def run_daily(cfg: Config) -> list[Prediction]:
    """CLI entry point: run for the watchlist.json symbols."""
    watchlist = load_watchlist_items(cfg)
    return run_for_symbols(watchlist, cfg)
