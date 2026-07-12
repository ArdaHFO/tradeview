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


def test_neutral_side_does_not_penalize_confidence():
    # A merely-neutral technical read must not be treated as "disagreeing" with
    # bullish news — confidence should be the plain weighted blend, not halved.
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.5, confidence=0.6,
                        rationale="positive")
    tech = TechnicalVerdict(symbol="AAPL", score=0.0, reasons=["flat"])
    pred = combine(news, tech, price=150.0, cfg=_cfg())
    assert pred.final_confidence == 0.3  # 0.5*0.6 + 0.5*0, no penalty


def test_zero_weight_side_ignored_in_confidence():
    # In technical_only, disagreeing news (which has zero weight) must not drag
    # the confidence down.
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.6, confidence=0.9,
                        rationale="bullish")
    tech = TechnicalVerdict(symbol="AAPL", score=-0.4, reasons=["downtrend"])
    pred = combine(news, tech, price=150.0, cfg=_cfg(), profile="technical_only")
    assert pred.final_direction == Direction.DOWN
    assert pred.final_confidence == 0.4  # 1.0*|-0.4|, news ignored


def test_unavailable_news_hands_weight_to_technicals():
    # No usable news signal (confidence 0) should not halve the technical read.
    news = NewsVerdict(symbol="AAPL", direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
                        rationale="NEWS_UNAVAILABLE (no Groq API key)")
    tech = TechnicalVerdict(symbol="AAPL", score=0.6, reasons=["uptrend"])
    pred = combine(news, tech, price=150.0, cfg=_cfg())
    assert pred.final_score == 0.6            # not 0.3
    assert pred.final_direction == Direction.UP
    assert pred.final_confidence == 0.6       # 1.0*|0.6|


def test_learned_profile_uses_model_score():
    # In "learned" mode the model's score drives the prediction directly,
    # regardless of the raw news/technical scores.
    news = NewsVerdict(symbol="AAPL", direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
                        rationale="n/a")
    tech = TechnicalVerdict(symbol="AAPL", score=-0.9, reasons=["bearish"])
    pred = combine(news, tech, price=100.0, cfg=_cfg(), profile="learned", learned_score=0.8)
    assert pred.final_score == 0.8
    assert pred.final_direction == Direction.UP
    assert pred.final_confidence == 0.8


def test_learned_profile_falls_back_to_blend_without_model():
    # profile "learned" but no learned_score (no model) -> behaves like balanced.
    news = NewsVerdict(symbol="AAPL", direction=Direction.UP, score=0.6, confidence=0.7,
                        rationale="bullish")
    tech = TechnicalVerdict(symbol="AAPL", score=0.4, reasons=["up"])
    pred = combine(news, tech, price=100.0, cfg=_cfg(), profile="learned", learned_score=None)
    assert abs(pred.final_score - 0.5) < 1e-9  # 0.5*0.6 + 0.5*0.4


def test_news_only_with_no_news_stays_neutral():
    # A news_only run with no news has nothing to fall back on — stays neutral.
    news = NewsVerdict(symbol="AAPL", direction=Direction.NEUTRAL, score=0.0, confidence=0.0,
                        rationale="no recent news found")
    tech = TechnicalVerdict(symbol="AAPL", score=0.8, reasons=["uptrend"])
    pred = combine(news, tech, price=150.0, cfg=_cfg(), profile="news_only")
    assert pred.final_score == 0.0
    assert pred.final_direction == Direction.NEUTRAL
