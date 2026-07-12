"""Orchestration: watchlist -> parallel news/technical -> fusion -> record."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .config import Config
from .fusion.combine import combine
from .learning.features import features_from_bars, market_index_for
from .learning.train import load_model
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


def _technical_pipeline(item: dict, cfg: Config) -> tuple[TechnicalVerdict, float | None, dict | None]:
    timeframe = str(item.get("timeframe", "1d"))
    bars = fetch_bars(item["symbol"], cfg, timeframe)
    verdict = score_technical(item["symbol"], bars)
    price = bars[-1].close if bars else None
    features = None
    if bars:
        market_bars = None
        if str(item.get("profile")) == "learned":
            # Regime / relative-strength features need the symbol's market index.
            try:
                market_bars = fetch_bars(market_index_for(item["symbol"]), cfg, timeframe)
            except Exception:
                market_bars = None
        features = features_from_bars(bars, market_bars=market_bars)  # news filled in later
    return verdict, price, features


def _analysis_meta(item: dict) -> tuple[str, str, str]:
    return (
        str(item.get("timeframe", "1d")),
        str(item.get("profile", "balanced")),
        ",".join(item.get("news_sources") or ["google"]),
    )


def _news_key(item: dict) -> str:
    # News content depends only on symbol + which sources were queried, not
    # on timeframe or profile — keying on those too (as before) meant a
    # 3-profile "compare" run fetched and re-analyzed identical articles
    # 3x, tripling Groq calls for no new information.
    news_sources = ",".join(item.get("news_sources") or ["google"])
    return f"{item['symbol']}|{news_sources}"


def _tech_key(item: dict) -> str:
    # Technical score depends on symbol + timeframe, not on profile.
    timeframe = str(item.get("timeframe", "1d"))
    return f"{item['symbol']}|{timeframe}"


def run_for_symbols(symbols: list[dict], cfg: Config, progress_cb=None, user_id: int | None = None) -> list[Prediction]:
    """Run the news+technical+fusion pipeline for an ad-hoc list of symbols.

    `symbols` is a list of {"symbol": ..., "name": ...} dicts (name optional).
    """
    def report(msg: str) -> None:
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    report("resolving prior predictions...")
    backfill.run(cfg, user_id=user_id)

    report(f"fetching news + technicals for {len(symbols)} symbol(s)...")
    news_items = {_news_key(item): item for item in symbols}
    tech_items = {_tech_key(item): item for item in symbols}
    # Cap concurrency: an unbounded 2x-per-run pool risks Yahoo/Groq 429s
    # once a compare/multi-timeframe run fans out to dozens of symbols.
    max_workers = min(8, max(2, len(news_items) + len(tech_items)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        news_futures = {key: ex.submit(_news_pipeline, item, cfg) for key, item in news_items.items()}
        tech_futures = {key: ex.submit(_technical_pipeline, item, cfg) for key, item in tech_items.items()}

    # Load the learned model once, only if some item actually asks for it.
    learned_model = None
    if any(str(item.get("profile")) == "learned" for item in symbols):
        learned_model = load_model(cfg.model_path)
        if learned_model is None:
            log.warning("learned profile requested but no usable model at %s — falling back to balanced blend", cfg.model_path)

    predictions: list[Prediction] = []
    recorder = PredictionRecorder(cfg.db_path)
    try:
        for item in symbols:
            symbol = item["symbol"]
            news_verdict = news_futures[_news_key(item)].result()
            tech_verdict, price, features = tech_futures[_tech_key(item)].result()
            if price is None:
                log.warning("skipping %s: no technical price data", symbol)
                continue
            timeframe, profile, news_sources = _analysis_meta(item)

            learned_score = None
            if profile == "learned" and learned_model is not None and features is not None:
                feat = dict(features)
                feat["news"] = news_verdict.score  # fill in the live news feature
                proba_up = learned_model.predict_from_dict(feat)
                learned_score = 2.0 * proba_up - 1.0

            prediction = combine(
                news_verdict,
                tech_verdict,
                price,
                cfg,
                timeframe=timeframe,
                profile=profile,
                news_sources=news_sources,
                name=item.get("name") or "",
                learned_score=learned_score,
            )
            recorder.record(prediction, user_id=user_id)
            predictions.append(prediction)
    finally:
        recorder.close()

    report("done.")
    return predictions


def run_daily(cfg: Config) -> list[Prediction]:
    """CLI entry point: run for the watchlist.json symbols."""
    watchlist = load_watchlist_items(cfg)
    return run_for_symbols(watchlist, cfg)
