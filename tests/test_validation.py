"""Tests for the statistical validation layer.

The point of these is that a validator which silently mis-reports "edge proven"
is worse than no validator at all, so the assertions pin down the cases where
the answer is knowable a priori: a pure coin flip must not pass, a strategy with
an overwhelming edge must, and the log parser must round-trip real output.
"""
from __future__ import annotations

import random

import pytest

from scanner.validation import (Trade, bootstrap, equity_curve, max_drawdown,
                                monte_carlo, profit_factor, trades_from_log,
                                validate, win_rate)


def _trade(r: float, pnl: float | None = None, setup: str = "VWAP_REVERSION",
           day: str = "2026-06-01") -> Trade:
    return Trade(day=day, time="15:00", symbol="TEST", setup=setup, side="LONG",
                 entry=10.0, exit=11.0, exit_reason="target", r_multiple=r,
                 pnl=pnl if pnl is not None else r * 50.0)


# --- point metrics --------------------------------------------------------

def test_profit_factor_basic():
    assert profit_factor([2.0, -1.0, 2.0, -1.0]) == pytest.approx(2.0)
    assert profit_factor([-1.0, -1.0]) == 0.0
    assert profit_factor([1.0, 2.0]) == float("inf")


def test_win_rate_and_equity_curve():
    assert win_rate([1.0, -1.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert equity_curve([10.0, -5.0, 20.0]) == [0.0, 10.0, 5.0, 25.0]


def test_max_drawdown_measures_peak_to_trough():
    # peak 100 -> trough 40 is the worst decline, even though it ends up higher
    curve = [0.0, 100.0, 40.0, 130.0]
    assert max_drawdown(curve) == pytest.approx(60.0)


def test_max_drawdown_zero_on_monotonic_rise():
    assert max_drawdown([0.0, 1.0, 2.0, 3.0]) == 0.0


# --- bootstrap ------------------------------------------------------------

def test_bootstrap_rejects_a_coin_flip():
    """A 50/50 +1R/-1R strategy has no edge; the CI must straddle zero."""
    rng = random.Random(7)
    trades = [_trade(1.0 if rng.random() < 0.5 else -1.0) for _ in range(200)]
    res = bootstrap(trades, iterations=2000, seed=1)
    assert res.expectancy_r.lo < 0 < res.expectancy_r.hi
    assert res.p_no_edge > 0.05          # "no edge" is a plausible reading


def test_bootstrap_accepts_an_overwhelming_edge():
    """+2R wins 80% of the time is unmistakable; the CI must clear zero."""
    trades = [_trade(2.0) for _ in range(80)] + [_trade(-1.0) for _ in range(20)]
    res = bootstrap(trades, iterations=2000, seed=1)
    assert res.expectancy_r.lo > 0
    assert res.p_no_edge < 0.01
    assert res.profit_factor.lo > 1.0


def test_bootstrap_interval_brackets_the_point_estimate():
    trades = [_trade(1.5), _trade(-1.0), _trade(0.8), _trade(-1.0), _trade(2.2)]
    res = bootstrap(trades, iterations=1000, seed=3)
    assert res.expectancy_r.lo <= res.expectancy_r.point <= res.expectancy_r.hi
    assert res.win_rate.lo <= res.win_rate.point <= res.win_rate.hi


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    trades = [_trade(1.0), _trade(-1.0), _trade(1.5), _trade(-0.5)]
    a = bootstrap(trades, iterations=500, seed=99)
    b = bootstrap(trades, iterations=500, seed=99)
    assert a.expectancy_r.lo == b.expectancy_r.lo
    assert a.expectancy_r.hi == b.expectancy_r.hi


def test_bootstrap_needs_two_trades():
    with pytest.raises(ValueError):
        bootstrap([_trade(1.0)])


def test_trades_needed_is_none_when_expectancy_is_negative():
    trades = [_trade(-1.0) for _ in range(10)] + [_trade(0.5) for _ in range(5)]
    res = bootstrap(trades, iterations=500, seed=1)
    assert res.trades_needed is None     # more data cannot rescue a negative edge


# --- monte carlo ----------------------------------------------------------

def test_monte_carlo_final_pnl_varies_under_resampling():
    """Resampling (unlike pure reordering) must produce a real PnL spread."""
    rng = random.Random(11)
    trades = [_trade(1.0 if rng.random() < 0.5 else -1.0) for _ in range(100)]
    mc = monte_carlo(trades, start_equity=10_000.0, iterations=2000, seed=5)
    assert mc.final_pnl.lo < mc.final_pnl.hi
    assert 0.0 < mc.p_losing_run < 1.0


def test_monte_carlo_reorder_band_is_tighter_than_resample_band():
    """Reordering holds the edge fixed, so its drawdown spread is narrower."""
    rng = random.Random(13)
    trades = [_trade(1.0 if rng.random() < 0.5 else -1.0) for _ in range(120)]
    mc = monte_carlo(trades, start_equity=10_000.0, iterations=2000, seed=5)
    reorder_span = mc.shuffle_max_dd.hi - mc.shuffle_max_dd.lo
    resample_span = mc.max_dd.hi - mc.max_dd.lo
    assert reorder_span < resample_span


def test_monte_carlo_all_winners_never_lose():
    trades = [_trade(1.0) for _ in range(30)]
    mc = monte_carlo(trades, start_equity=10_000.0, iterations=500, seed=5)
    assert mc.p_losing_run == 0.0
    assert mc.realized_max_dd == 0.0


# --- log parsing ----------------------------------------------------------

SAMPLE_LOG = """=== BACKTEST 2026-05-26 — 10 symbols, 2 trades ===
  15:00 AKTX   VWAP_REVERSION SHORT in    18.88 out    19.32 (stop  ) R -1.17  PnL $  -58.30
  19:20 AHTpF  VWAP_REVERSION LONG  in     6.08 out     6.29 (target) R +1.33  PnL $  +66.28
  ── win rate 50%  avg R +0.08  profit factor 1.14  total PnL $+7.98

=== BACKTEST 2026-05-27 — 10 symbols, 1 trades ===
  14:19 BRAI   GAP_AND_GO     LONG  in    13.30 out    12.52 (eod   ) R -1.40  PnL $-1,069.27
  ── win rate 0%  avg R -1.40  profit factor 0.00  total PnL $-1069.27
"""


def test_log_parser_extracts_trades_with_day_context(tmp_path):
    p = tmp_path / "bt.log"
    p.write_text(SAMPLE_LOG, encoding="utf-8")
    trades = trades_from_log(p)
    assert len(trades) == 3
    assert [t.day for t in trades] == ["2026-05-26", "2026-05-26", "2026-05-27"]
    assert [t.setup for t in trades] == ["VWAP_REVERSION", "VWAP_REVERSION",
                                         "GAP_AND_GO"]


def test_log_parser_handles_mixed_case_tickers_and_thousands_separator(tmp_path):
    p = tmp_path / "bt.log"
    p.write_text(SAMPLE_LOG, encoding="utf-8")
    trades = trades_from_log(p)
    assert trades[1].symbol == "AHTpF"        # preferred-share ticker
    assert trades[1].pnl == pytest.approx(66.28)
    assert trades[2].pnl == pytest.approx(-1069.27)   # comma stripped
    assert trades[2].exit_reason == "eod"


def test_log_parser_ignores_summary_lines(tmp_path):
    """Summary lines also contain 'R +x' and must not be mistaken for trades."""
    p = tmp_path / "bt.log"
    p.write_text(SAMPLE_LOG, encoding="utf-8")
    assert all(t.symbol in ("AKTX", "AHTpF", "BRAI") for t in trades_from_log(p))


# --- end to end -----------------------------------------------------------

def test_validate_reports_no_edge_for_a_coin_flip():
    rng = random.Random(17)
    trades = [_trade(1.0 if rng.random() < 0.5 else -1.0) for _ in range(150)]
    _, _, vd, report = validate(trades, start_equity=10_000.0, iterations=1000)
    assert not vd.passed
    assert "KANITLANMADI" in vd.headline
    assert "İSTATİSTİKSEL DOĞRULAMA" in report


def test_validate_passes_a_strong_edge():
    trades = [_trade(2.0) for _ in range(90)] + [_trade(-1.0) for _ in range(30)]
    _, _, vd, _ = validate(trades, start_equity=10_000.0, iterations=1000)
    assert vd.passed


def test_validate_calls_out_a_proven_loser_distinctly():
    """A CI entirely below zero is a systematic loss, not an unproven edge.

    Conflating the two would tell us to gather more data on a setup that should
    be retired instead.
    """
    trades = [_trade(-1.0) for _ in range(80)] + [_trade(0.5) for _ in range(20)]
    boot, _, vd, report = validate(trades, start_equity=10_000.0, iterations=2000)
    assert boot.expectancy_r.hi < 0
    assert not vd.passed
    assert "ZARARDA" in vd.headline
    assert "KANITLANMADI" not in vd.headline
    assert any("sistematik" in n for n in vd.notes)
    assert "sıfırı içeriyor" not in report      # the straddle wording must not leak
