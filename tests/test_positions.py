"""Tests for open-position tracking (the SELL side).

The exit rules must match the backtester's exactly, including their precedence:
the measured edge assumes positions close the way the backtest closed them, so a
tracker that exits on a different condition, or checks them in a different
order, silently invalidates every number the strategy was accepted on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.swing.positions import (Position, exit_level, format_review,
                                     load_positions, review, save_positions,
                                     status)
from scanner.swing.strategies import MeanReversion, build


def _frame(closes, highs=None, lows=None, start="2026-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    return pd.DataFrame({
        "Open": closes,
        "High": np.asarray(highs, dtype=float) if highs is not None else closes + 1,
        "Low": np.asarray(lows, dtype=float) if lows is not None else closes - 1,
        "Close": closes,
        "Volume": np.full(n, 1e9),
    }, index=pd.bdate_range(start, periods=n))


def _pos(entry=100.0, stop=90.0, entry_date="2026-01-01") -> Position:
    return Position(symbol="AAA", strategy="meanrev", entry_date=entry_date,
                    entry_price=entry, shares=10, stop=stop)


# --- persistence ----------------------------------------------------------

def test_positions_round_trip(tmp_path):
    path = tmp_path / "positions.json"
    save_positions([_pos(), _pos(entry=50.0)], path)
    back = load_positions(path)
    assert len(back) == 2
    assert back[0].symbol == "AAA"
    assert back[1].entry_price == 50.0


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert load_positions(tmp_path / "nope.json") == []


# --- exit precedence ------------------------------------------------------

def test_stop_takes_priority_over_everything():
    """Price under the stop must read SELL even if other rules also fire."""
    df = _frame([100.0] * 40 + [80.0])
    st = status(_pos(stop=90.0), df, MeanReversion())
    assert st.action == "SELL"
    assert "stop" in st.reason


def test_signal_exit_fires_when_close_reclaims_the_mean():
    # a dip then a sharp recovery pushes close above SMA5
    closes = [100.0] * 40 + [90.0, 90.0, 90.0, 90.0, 110.0]
    st = status(_pos(stop=50.0), _frame(closes), MeanReversion())
    assert st.action == "SELL"
    assert "sma5" in st.reason


def test_time_stop_fires_after_max_hold():
    """Flat price: no stop, no signal -- only the clock closes it."""
    closes = list(np.linspace(100.0, 100.5, 60))     # drifts up, never below SMA5
    df = _frame(closes)
    st = status(_pos(stop=50.0, entry_date="2026-01-01"), df, MeanReversion())
    assert st.bars_held >= MeanReversion.max_hold_days
    assert st.action == "SELL"


def test_hold_when_no_rule_fires():
    closes = [100.0] * 30 + [95.0]        # below SMA5, above stop, recently opened
    st = status(_pos(stop=80.0, entry_date=str(_frame(closes).index[-2].date())),
                _frame(closes), MeanReversion())
    assert st.action == "HOLD"


# --- reported numbers -----------------------------------------------------

def test_unrealized_pnl_and_r_multiple():
    df = _frame([100.0] * 30 + [110.0])
    st = status(_pos(entry=100.0, stop=90.0), df, MeanReversion())
    assert st.unrealized_pnl == pytest.approx(100.0)     # +10 x 10 shares
    assert st.unrealized_r == pytest.approx(1.0)         # +10 against 10 risk


def test_r_multiple_is_zero_when_risk_is_degenerate():
    """A stop at or above entry has no meaningful R; must not divide by <= 0."""
    df = _frame([100.0] * 30)
    st = status(_pos(entry=100.0, stop=120.0), df, MeanReversion())
    assert st.unrealized_r == 0.0


def test_exit_level_reports_the_live_rule_price():
    df = _frame([100.0] * 30)
    ind_sma5 = 100.0
    assert exit_level("sma5", _frame([100.0] * 30).assign(
        sma5=ind_sma5).iloc[-1]) == pytest.approx(ind_sma5)
    st = status(_pos(), df, MeanReversion())
    assert st.exit_level == pytest.approx(100.0)


def test_unknown_exit_rule_has_no_level():
    row = pd.Series({"sma5": 10.0})
    assert exit_level("nonexistent", row) is None


# --- review ---------------------------------------------------------------

def test_review_lists_sells_before_holds():
    frames = {"SELLME": _frame([100.0] * 40 + [10.0]),
              "HOLDME": _frame([100.0] * 30 + [95.0])}
    positions = [
        Position("HOLDME", "meanrev", "2026-02-20", 100.0, 10, 80.0),
        Position("SELLME", "meanrev", "2026-02-20", 100.0, 10, 90.0),
    ]
    rows = review(positions, frames, build)
    assert [r.position.symbol for r in rows] == ["SELLME", "HOLDME"]


def test_review_skips_symbols_without_data():
    rows = review([_pos()], {}, build)
    assert rows == []


def test_format_review_prompts_when_empty():
    text = format_review([])
    assert "kayıtlı pozisyon yok" in text
    assert "--add" in text


def test_format_review_warns_about_next_open_when_a_sell_exists():
    frames = {"AAA": _frame([100.0] * 40 + [10.0])}
    rows = review([_pos(stop=90.0)], frames, build)
    text = format_review(rows)
    assert "SAT" in text
    assert "AÇILIŞINDA" in text.upper()
