"""Configuration: env-driven, with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass
class Config:
    groq_api_key: str = field(
        default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    db_path: str = field(
        default_factory=lambda: os.environ.get("DATABASE_URL") or os.environ.get("FORECASTER_DB_PATH", "predictions.db"))
    watchlist_path: str = field(
        default_factory=lambda: os.environ.get("WATCHLIST_PATH", "watchlist.json"))
    registration_code: str = field(
        default_factory=lambda: os.environ.get("REGISTRATION_CODE", ""))
    cookie_secure: bool = field(
        default_factory=lambda: os.environ.get("COOKIE_SECURE", "true").strip().lower() != "false")

    news_lookback_hours: int = field(
        default_factory=lambda: _env_int("NEWS_LOOKBACK_HOURS", 24))
    max_articles_per_symbol: int = field(
        default_factory=lambda: _env_int("MAX_ARTICLES_PER_SYMBOL", 10))
    dedupe_similarity_threshold: float = 0.85

    intraday_lookback_period: str = field(
        default_factory=lambda: os.environ.get("INTRADAY_LOOKBACK_PERIOD", "60d"))

    news_weight: float = field(
        default_factory=lambda: _env_float("NEWS_WEIGHT", 0.5))
    technical_weight: float = field(
        default_factory=lambda: _env_float("TECHNICAL_WEIGHT", 0.5))
    neutral_band: float = field(
        default_factory=lambda: _env_float("NEUTRAL_BAND", 0.15))

    groq_model: str = field(
        default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))

    technical_lookback_period: str = "6mo"
    max_symbols_per_run: int = field(
        default_factory=lambda: _env_int("MAX_SYMBOLS_PER_RUN", 10))


def load_config() -> Config:
    """Load .env (if python-dotenv is installed) then build Config from env."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Config()
