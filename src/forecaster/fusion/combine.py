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

    # If the news side produced no usable signal (no Groq key, no articles, or a
    # genuinely no-bearing read → confidence 0), don't let it dilute the blend
    # toward neutral: hand its weight to the technical side so the prediction
    # rests on the information we actually have. Only when the profile still
    # gives technicals a say — a pure "news_only" run with no news stays neutral.
    nw, tw = news_weight, technical_weight
    if news.confidence <= 0.0 and technical_weight > 0:
        nw, tw = 0.0, news_weight + technical_weight

    final_score = nw * news.score + tw * technical.score

    if abs(final_score) < neutral_band:
        direction = Direction.NEUTRAL
    elif final_score > 0:
        direction = Direction.UP
    else:
        direction = Direction.DOWN

    news_sign = 1 if news.score > 0 else (-1 if news.score < 0 else 0)
    tech_sign = 1 if technical.score > 0 else (-1 if technical.score < 0 else 0)

    # Confidence is a weight-aware blend of each side's own conviction, so a side
    # with no weight in this profile can't drive it. The agreement bonus /
    # disagreement penalty applies only when BOTH sides carry weight AND give a
    # directional (non-neutral) read — a merely-neutral side used to be mistaken
    # for "disagreement" and wrongly halve the confidence.
    base_confidence = nw * news.confidence + tw * abs(technical.score)
    both_directional = nw > 0 and tw > 0 and news_sign != 0 and tech_sign != 0
    if both_directional and news_sign == tech_sign:
        final_confidence = min(1.0, base_confidence * 1.2)
    elif both_directional and news_sign != tech_sign:
        final_confidence = base_confidence * 0.6
    else:
        final_confidence = base_confidence

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
