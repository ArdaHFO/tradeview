"""Risk manager: ATR-based sizing, daily kill-switch, PDT counter, session lockout."""
from __future__ import annotations

from datetime import date, datetime

from .config import RiskConfig
from .session import in_entry_window, to_ny


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.daily_pnl = 0.0
        self._day: date | None = None
        self._day_trades: list[date] = []     # one entry per round-trip day trade

    # ---- lifecycle ------------------------------------------------------

    def on_new_session(self, dt: datetime) -> None:
        today = to_ny(dt).date()
        if self._day != today:
            self._day = today
            self.daily_pnl = 0.0

    def on_fill_pnl(self, pnl: float) -> None:
        """Call when a round-trip closes (paper or live)."""
        self.daily_pnl += pnl

    def register_day_trade(self, dt: datetime) -> None:
        self._day_trades.append(to_ny(dt).date())

    # ---- gates ----------------------------------------------------------

    def kill_switch_active(self) -> bool:
        limit = -self.cfg.equity * self.cfg.daily_kill_switch_pct / 100.0
        return self.daily_pnl <= limit

    def day_trades_used_5d(self, dt: datetime) -> int:
        today = to_ny(dt).date()
        return sum(1 for d in self._day_trades if (today - d).days < 5)

    def pdt_blocked(self, dt: datetime) -> bool:
        if not self.cfg.pdt_account:
            return False
        return self.day_trades_used_5d(dt) >= self.cfg.max_day_trades_per_5d

    def can_enter(self, dt: datetime) -> tuple[bool, str]:
        """All entry gates in one call. Returns (allowed, reason_if_blocked)."""
        if not in_entry_window(dt, self.cfg.no_entry_last_minutes):
            return False, "outside entry window (closed or last-15-min lockout)"
        if self.kill_switch_active():
            return False, f"daily kill-switch hit ({self.daily_pnl:+.2f} USD)"
        if self.pdt_blocked(dt):
            return False, "PDT limit reached (3 day-trades / 5 days)"
        return True, ""

    # ---- sizing ---------------------------------------------------------

    def position_size(self, entry: float, stop: float) -> tuple[int, float]:
        """Shares and dollar risk for the configured per-trade risk %.

        size = floor(equity * risk% / |entry - stop|), also capped so the
        position notional never exceeds equity (no leverage in v1).
        """
        per_share = abs(entry - stop)
        if per_share <= 0:
            return 0, 0.0
        risk_dollars = self.cfg.equity * self.cfg.risk_per_trade_pct / 100.0
        shares = int(risk_dollars / per_share)
        if entry > 0:
            max_shares = int(self.cfg.equity / entry)
            shares = min(shares, max_shares)
        return max(shares, 0), round(shares * per_share, 2)
