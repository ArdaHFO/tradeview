"""Combine a NewsVerdict and TechnicalVerdict into a final Prediction."""
from __future__ import annotations

from datetime import datetime, timezone

from ..config import Config
from ..models import Direction, NewsVerdict, Prediction, TechnicalVerdict


def combine(news: NewsVerdict, technical: TechnicalVerdict, price: float, cfg: Config) -> Prediction:
    final_score = cfg.news_weight * news.score + cfg.technical_weight * technical.score

    if abs(final_score) < cfg.neutral_band:
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
        news_score=news.score,
        news_confidence=news.confidence,
        news_rationale=news.rationale,
        technical_score=technical.score,
        technical_reasons=technical.reasons,
        final_score=final_score,
        final_direction=direction,
        final_confidence=round(final_confidence, 3),
        price_at_prediction=price,
    )
