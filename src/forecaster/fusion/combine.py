"""Combine a NewsVerdict and TechnicalVerdict into a final Prediction."""
from __future__ import annotations

from datetime import datetime, timezone

from ..config import Config
from ..models import Direction, NewsVerdict, Prediction, TechnicalVerdict


def _profile_weights(cfg: Config, profile: str) -> tuple[float, float, float]:
    profiles: dict[str, tuple[float, float, float]] = {
        "balanced": (cfg.news_weight, cfg.technical_weight, cfg.neutral_band),
        "news_heavy": (0.7, 0.3, 0.12),
        "technical_heavy": (0.3, 0.7, 0.12),
        "news_only": (1.0, 0.0, 0.10),
        "technical_only": (0.0, 1.0, 0.10),
    }
    return profiles.get(profile, profiles["balanced"])


def combine(
    news: NewsVerdict,
    technical: TechnicalVerdict,
    price: float,
    cfg: Config,
    *,
    timeframe: str = "1d",
    profile: str = "balanced",
    news_sources: str = "google",
    name: str = "",
) -> Prediction:
    news_weight, technical_weight, neutral_band = _profile_weights(cfg, profile)
    final_score = news_weight * news.score + technical_weight * technical.score

    if abs(final_score) < neutral_band:
        direction = Direction.NEUTRAL
    elif final_score > 0:
        direction = Direction.UP
    else:
        direction = Direction.DOWN

    news_sign = 1 if news.score > 0 else (-1 if news.score < 0 else 0)
    tech_sign = 1 if technical.score > 0 else (-1 if technical.score < 0 else 0)
    agree = news_sign != 0 and news_sign == tech_sign
    avg_confidence = (news.confidence + abs(technical.score)) / 2
    final_confidence = min(1.0, avg_confidence * 1.2) if agree else avg_confidence * 0.6

    return Prediction(
        ts=datetime.now(timezone.utc),
        symbol=news.symbol,
        name=name,
        timeframe=timeframe,
        profile=profile,
        news_sources=news_sources,
        news_score=news.score,
        news_confidence=news.confidence,
        news_rationale=news.rationale,
        technical_score=technical.score,
        technical_reasons=technical.reasons,
        technical_indicators=technical.indicators,
        final_score=final_score,
        final_direction=direction,
        final_confidence=round(final_confidence, 3),
        price_at_prediction=price,
    )
