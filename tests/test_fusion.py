from forecaster.config import Config
from forecaster.fusion.combine import combine
from forecaster.models import Direction, NewsVerdict, TechnicalVerdict


def _cfg(**overrides) -> Config:
    cfg = Config(groq_api_key="x")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_agreement_boosts_confidence():
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.6, confidence=0.7,
                        rationale="good earnings")
    tech = TechnicalVerdict(symbol="AAPL", score=0.5, reasons=["uptrend"])
    cfg = _cfg()
    pred = combine(news, tech, price=150.0, cfg=cfg)
    assert pred.final_direction == Direction.UP
    assert pred.final_confidence > (news.confidence + abs(tech.score)) / 2


def test_conflict_lowers_confidence():
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.6, confidence=0.7,
                        rationale="good earnings")
    tech = TechnicalVerdict(symbol="AAPL", score=-0.5, reasons=["downtrend"])
    cfg = _cfg()
    pred = combine(news, tech, price=150.0, cfg=cfg)
    assert pred.final_confidence < (news.confidence + abs(tech.score)) / 2


def test_small_combined_score_is_neutral():
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.05, confidence=0.2,
                        rationale="minor news")
    tech = TechnicalVerdict(symbol="AAPL", score=0.05, reasons=["flat"])
    cfg = _cfg(neutral_band=0.15)
    pred = combine(news, tech, price=150.0, cfg=cfg)
    assert pred.final_direction == Direction.NEUTRAL
