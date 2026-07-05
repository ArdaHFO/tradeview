"""Configuration: env-driven, with the locked product defaults as fallbacks."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class ScreenerConfig:
    """Stage-1 filters. Defaults target small/mid-float momentum names."""
    min_rvol: float = 2.0
    min_abs_gap_pct: float = 4.0
    # Backtest finding (2026-06): sub-$5 thin names lose to spread/slippage
    # (GAP_AND_GO 26% win there vs 67% on liquid $15+ names). Stay liquid.
    min_price: float = 5.0
    max_price: float = 50.0
    max_float_shares: float = 50_000_000
    min_day_range_pct: float = 4.0
    max_watchlist_size: int = 30
    require_float: bool = False     # if False, unknown float passes (with score penalty)


@dataclass
class SignalConfig:
    """Stage-2 confluence scoring."""
    threshold: float = 0.65
    # Family weights must sum to 1.0; order flow carries the edge.
    w_orderflow: float = 0.35
    w_position: float = 0.25
    w_structure: float = 0.25
    w_volatility: float = 0.15
    target_r_multiple: float = 2.0       # default take-profit at 2R
    slippage_bps: float = 20.0           # per-side slippage in basis points (backtest)
    cooldown_minutes: int = 30           # per symbol+setup re-alert lockout
    min_stop_slippage_mult: float = 10.0  # skip if |entry-stop| < N * one-side slippage
    # GAP_AND_GO and ORB disabled by default: 15-day backtest showed consistent
    # losses for both (PF 0.35-0.46 across two parameter-fix attempts each --
    # wider stops/margins and hard order-flow gates failed to make either
    # profitable). Only VWAP_REVERSION cleared PF > 1 in the same window.
    disabled_setups: str = "GAP_AND_GO,ORB"  # comma-separated SetupType names to turn off
    vwap_reversion_atr_dist: float = 2.0  # min ATRs from VWAP to consider a fade
    # Widened from 15 -- backtest showed the 15-min range was too noisy/unstable,
    # causing frequent false ORB breakouts.
    opening_range_minutes: int = 30
    large_print_multiple: float = 10.0    # trade size >= 10x avg size = large print


@dataclass
class RiskConfig:
    equity: float = field(default_factory=lambda: _env_float("ACCOUNT_EQUITY", 10_000.0))
    risk_per_trade_pct: float = field(
        default_factory=lambda: _env_float("RISK_PER_TRADE_PCT", 0.5))
    daily_kill_switch_pct: float = field(
        default_factory=lambda: _env_float("DAILY_KILL_SWITCH_PCT", 2.0))
    pdt_account: bool = field(default_factory=lambda: _env_bool("PDT_ACCOUNT", True))
    max_day_trades_per_5d: int = 3       # PDT limit for accounts under $25k
    no_entry_last_minutes: int = 15      # no new entries in the last N minutes of RTH


@dataclass
class Config:
    polygon_api_key: str = field(
        default_factory=lambda: os.environ.get("POLYGON_API_KEY", ""))
    telegram_bot_token: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(
        default_factory=lambda: os.environ.get("TELEGRAM_CHAT_ID", ""))
    db_path: str = field(
        default_factory=lambda: os.environ.get("SCANNER_DB_PATH", "signals.db"))
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config() -> Config:
    """Load .env (if python-dotenv is installed) then build Config from env."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return Config()
