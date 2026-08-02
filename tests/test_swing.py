"""Tests for the daily-bar swing engine.

The assertions concentrate on the ways a daily backtest silently lies:
filling at a price that was not yet knowable, resolving an ambiguous intraday
sequence in the strategy's favour, or pricing a gap-through-stop at the stop.
Each of those inflates results without ever throwing an error, so they are
pinned down explicitly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scanner.swing.backtest import SwingConfig, run
from scanner.swing.strategies import (Breakout, MeanReversion, Strategy,
                                      TrendPullback, add_indicators, atr, rsi)


def _frame(closes, highs=None, lows=None, opens=None, volume=1e9,
           start="2020-01-01") -> pd.DataFrame:
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "Open": np.asarray(opens, dtype=float) if opens is not None else closes,
        "High": np.asarray(highs, dtype=float) if highs is not None else closes,
        "Low": np.asarray(lows, dtype=float) if lows is not None else closes,
        "Close": closes,
        "Volume": np.full(n, volume),
    }, index=pd.bdate_range(start, periods=n))


# --- indicators -----------------------------------------------------------

def test_rsi_is_100_when_price_only_rises():
    r = rsi(pd.Series(np.arange(1, 40, dtype=float)), 2)
    assert r.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_near_zero_when_price_only_falls():
    r = rsi(pd.Series(np.arange(40, 1, -1, dtype=float)), 2)
    assert r.iloc[-1] < 1.0


def test_atr_tracks_a_constant_true_range():
    df = _frame(closes=[10] * 30, highs=[11] * 30, lows=[9] * 30)
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0, abs=0.01)


def test_donchian_high_excludes_today():
    """A channel that included today could never be broken -- it would be today."""
    closes = list(np.linspace(10, 20, 80))
    df = add_indicators(_frame(closes))
    assert df["dc_high55"].iloc[-1] < df["High"].iloc[-1]


# --- signal correctness ---------------------------------------------------

def test_meanrev_requires_both_uptrend_and_oversold():
    # steady downtrend: deeply oversold but below the 200MA, so no entry
    closes = list(np.linspace(100, 40, 300))
    df = add_indicators(_frame(closes))
    assert MeanReversion().entries(df).empty


def test_meanrev_fires_on_a_dip_inside_an_uptrend():
    closes = list(np.linspace(50, 150, 300))
    closes[-1] = closes[-2] * 0.90            # sharp one-day dip
    df = add_indicators(_frame(closes))
    ent = MeanReversion().entries(df)
    assert not ent.empty
    assert ent.index[-1] == df.index[-1]


def test_breakout_only_fires_on_the_first_bar_above_the_channel():
    """Every subsequent day is also above the channel; only the first is an entry.

    The base must actually consolidate: in a monotonic rise every close is a new
    high, so nothing is ever a *fresh* breakout.
    """
    base = list(50.0 + 3.0 * np.sin(np.arange(250) / 4.0))    # range-bound base
    closes = base + [80.0] * 5                                # then a clean break
    df = add_indicators(_frame(closes))
    ent = Breakout().entries(df)
    assert len(ent) == 1


def test_trend_pullback_needs_a_dip_then_a_reclaim():
    closes = list(np.linspace(50, 150, 300))
    df = add_indicators(_frame(closes))
    # a pure monotonic rise never dips below the 20MA, so nothing should fire
    assert TrendPullback().entries(df).empty


# --- fill discipline ------------------------------------------------------

SIGNAL_BAR = 20        # past ATR(14) warmup, so the fill is not skipped


class AlwaysDayTwenty(Strategy):
    """Fires once, on a fixed bar, so fill mechanics can be asserted exactly."""
    name = "fixture"
    stop_atr = 1.0
    max_hold_days = 5
    exit_rule = "none"

    def entries(self, df):
        mask = pd.Series(False, index=df.index)
        mask.iloc[SIGNAL_BAR] = True
        return self._frame(mask, pd.Series(1.0, index=df.index),
                           pd.Series("test", index=df.index))


def _flat(n: int = 40) -> pd.DataFrame:
    """Quiet series with a real high/low range, so ATR is well defined."""
    return _frame([100.0] * n, highs=[101.0] * n, lows=[99.0] * n,
                  opens=[100.0] * n)


def test_entry_fills_at_the_next_session_open_not_the_signal_close():
    df = _flat()
    df.iloc[SIGNAL_BAR + 1, df.columns.get_loc("Open")] = 105.0
    rep = run({"AAA": df}, AlwaysDayTwenty(), SwingConfig(slippage_bps=0.0,
                                                          commission=0.0))
    assert len(rep.trades) == 1
    assert rep.trades[0].entry == pytest.approx(105.0)   # not the 100.0 close


def test_gap_through_the_stop_fills_at_the_open_not_the_stop():
    """Pricing a gap at the stop price is the classic way to fake a good backtest."""
    df = _flat()
    gap = SIGNAL_BAR + 2                       # a bar after the entry
    for col, val in (("Open", 80.0), ("High", 81.0), ("Low", 79.0),
                     ("Close", 80.0)):
        df.iloc[gap, df.columns.get_loc(col)] = val
    rep = run({"AAA": df}, AlwaysDayTwenty(), SwingConfig(slippage_bps=0.0,
                                                          commission=0.0))
    assert rep.trades[0].exit_reason == "gap_stop"
    assert rep.trades[0].exit == pytest.approx(80.0)     # not the stop level
    assert rep.trades[0].pnl < 0


def test_stop_is_taken_when_the_low_pierces_it():
    df = _flat()
    hit = SIGNAL_BAR + 2
    df.iloc[hit, df.columns.get_loc("Low")] = 90.0       # deep enough for a 1-ATR stop
    rep = run({"AAA": df}, AlwaysDayTwenty(), SwingConfig(slippage_bps=0.0,
                                                          commission=0.0))
    assert rep.trades[0].exit_reason == "stop"


def test_time_stop_closes_the_position():
    rep = run({"AAA": _flat()}, AlwaysDayTwenty(),
              SwingConfig(slippage_bps=0.0, commission=0.0))
    assert rep.trades[0].exit_reason in ("time", "eod")


# --- portfolio constraints ------------------------------------------------

def test_position_slots_are_respected_and_reported():
    frames = {f"S{i}": _flat() for i in range(8)}
    rep = run(frames, AlwaysDayTwenty(), SwingConfig(max_positions=3,
                                                     slippage_bps=0.0,
                                                     commission=0.0))
    assert len(rep.trades) == 3
    assert rep.skipped_no_slot == 5
    assert any("slot" in w for w in rep.warnings)


def test_illiquid_symbols_are_filtered_out():
    df = _frame([100.0] * 40, highs=[101.0] * 40, lows=[99.0] * 40, volume=10.0)
    rep = run({"AAA": df}, AlwaysDayTwenty(),
              SwingConfig(min_dollar_volume=5_000_000.0))
    assert rep.trades == []


def test_survivorship_bias_is_always_disclosed():
    rep = run({"AAA": _flat()}, AlwaysDayTwenty(), SwingConfig())
    assert any("survivorship" in w.lower() for w in rep.warnings)


def test_trades_are_validation_compatible():
    """Swing output must feed the same bootstrap as the intraday backtester."""
    from scanner.validation import bootstrap
    frames = {f"S{i}": _flat() for i in range(6)}
    rep = run(frames, AlwaysDayTwenty(), SwingConfig(max_positions=6))
    assert len(rep.trades) >= 2
    res = bootstrap(rep.trades, iterations=200)
    assert res.n_trades == len(rep.trades)
