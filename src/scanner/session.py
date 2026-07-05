"""US equity session logic (America/New_York), DST-safe via zoneinfo."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
PREMARKET_OPEN = time(4, 0)


def to_ny(dt: datetime) -> datetime:
    """Convert any aware datetime to New York time. Naive input is assumed UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(NY)


def is_trading_day(dt: datetime) -> bool:
    """Weekday check only. NYSE holidays are not modeled in v1 -- the data feed
    simply produces nothing on holidays, so the engine stays idle anyway."""
    return to_ny(dt).weekday() < 5


def is_rth(dt: datetime) -> bool:
    """True during regular trading hours 09:30-16:00 ET on a weekday."""
    ny = to_ny(dt)
    return is_trading_day(dt) and RTH_OPEN <= ny.time() < RTH_CLOSE


def is_premarket(dt: datetime) -> bool:
    ny = to_ny(dt)
    return is_trading_day(dt) and PREMARKET_OPEN <= ny.time() < RTH_OPEN


def minutes_since_open(dt: datetime) -> float:
    """Minutes elapsed since today's 09:30 ET open (negative before open)."""
    ny = to_ny(dt)
    open_dt = ny.replace(hour=RTH_OPEN.hour, minute=RTH_OPEN.minute,
                         second=0, microsecond=0)
    return (ny - open_dt).total_seconds() / 60.0


def minutes_to_close(dt: datetime) -> float:
    ny = to_ny(dt)
    close_dt = ny.replace(hour=RTH_CLOSE.hour, minute=RTH_CLOSE.minute,
                          second=0, microsecond=0)
    return (close_dt - ny).total_seconds() / 60.0


def in_entry_window(dt: datetime, no_entry_last_minutes: int = 15) -> bool:
    """RTH and not inside the end-of-day lockout."""
    return is_rth(dt) and minutes_to_close(dt) > no_entry_last_minutes


def is_opening_drive(dt: datetime, window_minutes: int = 60) -> bool:
    """First N minutes after the open -- the gap-and-go window."""
    m = minutes_since_open(dt)
    return is_rth(dt) and 0 <= m <= window_minutes


def session_elapsed_fraction(dt: datetime) -> float:
    """Fraction of the RTH session elapsed, clamped to [0, 1].

    Used to prorate RVOL intraday. Linear proration is a v1 simplification
    (real volume is U-shaped); it overstates RVOL slightly mid-day, which only
    makes the screener marginally stricter mid-day -- an acceptable bias.
    """
    total = (RTH_CLOSE.hour * 60 + RTH_CLOSE.minute) - (RTH_OPEN.hour * 60 + RTH_OPEN.minute)
    m = minutes_since_open(dt)
    return max(0.0, min(1.0, m / total))


def rth_open_dt(dt: datetime) -> datetime:
    """Today's 09:30 ET as an aware datetime (NY tz)."""
    ny = to_ny(dt)
    return ny.replace(hour=RTH_OPEN.hour, minute=RTH_OPEN.minute,
                      second=0, microsecond=0)
