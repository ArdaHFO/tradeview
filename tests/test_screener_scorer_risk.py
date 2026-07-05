from datetime import datetime

import pytest

from scanner.config import RiskConfig, ScreenerConfig, SignalConfig
from scanner.models import (FamilyScores, SetupCandidate, SetupType, Side,
                            SymbolSnapshot)
from scanner.risk import RiskManager
from scanner.signal.scorer import ConfluenceScorer
from scanner.stage1_screener import screen
from scanner.session import NY


def snap(**kw):
    base = dict(symbol="TEST", price=10.0, prev_close=9.0,
                day_volume=5_000_000, avg_daily_volume=1_000_000,
                day_high=10.5, day_low=9.4, float_shares=20_000_000)
    base.update(kw)
    return SymbolSnapshot(**base)


class TestScreener:
    CFG = ScreenerConfig()

    def test_hot_symbol_passes(self):
        out = screen([snap()], self.CFG, elapsed_fraction=0.5)
        assert len(out) == 1
        assert out[0].symbol == "TEST"
        assert out[0].gap_pct == pytest.approx(11.11, abs=0.01)

    def test_price_bounds(self):
        assert not screen([snap(price=1.0, prev_close=0.9)], self.CFG, 0.5)
        assert not screen([snap(price=100.0, prev_close=90.0)], self.CFG, 0.5)

    def test_small_gap_filtered(self):
        assert not screen([snap(price=9.2, prev_close=9.0,
                                day_high=9.3, day_low=9.0)], self.CFG, 0.5)

    def test_low_rvol_filtered(self):
        assert not screen([snap(day_volume=100_000)], self.CFG, 0.5)

    def test_big_float_filtered(self):
        assert not screen([snap(float_shares=900_000_000)], self.CFG, 0.5)

    def test_unknown_float_passes_by_default(self):
        assert screen([snap(float_shares=None)], self.CFG, 0.5)

    def test_watchlist_capped_and_sorted(self):
        snaps = [snap(symbol=f"S{i}", day_volume=2_000_000 + i * 500_000)
                 for i in range(40)]
        out = screen(snaps, self.CFG, 0.5)
        assert len(out) == self.CFG.max_watchlist_size
        assert out[0].rvol >= out[-1].rvol


class TestScorer:
    def test_weights_must_sum_to_one(self):
        bad = SignalConfig(w_orderflow=0.5, w_position=0.5,
                           w_structure=0.5, w_volatility=0.5)
        with pytest.raises(ValueError):
            ConfluenceScorer(bad)

    def _cand(self, scores):
        return SetupCandidate(symbol="X", setup=SetupType.ORB, side=Side.LONG,
                              entry=10.0, stop=9.5, scores=scores)

    def test_threshold_gate(self):
        cfg = SignalConfig()
        scorer = ConfluenceScorer(cfg)
        now = datetime(2026, 6, 11, 10, 0, tzinfo=NY)
        weak = self._cand(FamilyScores(0.3, 0.3, 0.3, 0.3))
        assert scorer.score(weak, now, 100, 50.0) is None
        strong = self._cand(FamilyScores(0.9, 0.9, 0.9, 0.9))
        sig = scorer.score(strong, now, 100, 50.0)
        assert sig is not None
        assert sig.confluence == pytest.approx(0.9)
        # 2R target on a 0.5 risk: 10.0 + 1.0
        assert sig.target == pytest.approx(11.0)

    def test_short_target_below_entry(self):
        cfg = SignalConfig()
        scorer = ConfluenceScorer(cfg)
        now = datetime(2026, 6, 11, 10, 0, tzinfo=NY)
        cand = SetupCandidate(symbol="X", setup=SetupType.ORB, side=Side.SHORT,
                              entry=10.0, stop=10.5,
                              scores=FamilyScores(1, 1, 1, 1))
        sig = scorer.score(cand, now, 100, 50.0)
        assert sig.target == pytest.approx(9.0)


class TestRisk:
    def test_position_size(self):
        rm = RiskManager(RiskConfig(equity=10_000, risk_per_trade_pct=0.5))
        shares, risk = rm.position_size(entry=10.0, stop=9.5)
        assert shares == 100            # $50 risk / $0.5 per share
        assert risk == pytest.approx(50.0)

    def test_size_capped_by_equity(self):
        rm = RiskManager(RiskConfig(equity=10_000, risk_per_trade_pct=0.5))
        shares, _ = rm.position_size(entry=100.0, stop=99.9)   # tight stop
        assert shares <= 100            # notional cap: 10k / $100

    def test_zero_stop_distance(self):
        rm = RiskManager(RiskConfig())
        assert rm.position_size(10.0, 10.0) == (0, 0.0)

    def test_kill_switch(self):
        rm = RiskManager(RiskConfig(equity=10_000, daily_kill_switch_pct=2.0))
        rm.on_fill_pnl(-150.0)
        assert not rm.kill_switch_active()
        rm.on_fill_pnl(-60.0)
        assert rm.kill_switch_active()

    def test_pdt_counter(self):
        rm = RiskManager(RiskConfig(pdt_account=True))
        dt = datetime(2026, 6, 11, 10, 0, tzinfo=NY)
        for _ in range(3):
            rm.register_day_trade(dt)
        assert rm.pdt_blocked(dt)
        non_pdt = RiskManager(RiskConfig(pdt_account=False))
        for _ in range(5):
            non_pdt.register_day_trade(dt)
        assert not non_pdt.pdt_blocked(dt)

    def test_entry_gates(self):
        rm = RiskManager(RiskConfig())
        ok, _ = rm.can_enter(datetime(2026, 6, 11, 10, 0, tzinfo=NY))
        assert ok
        # last 15 minutes -> blocked
        blocked, why = rm.can_enter(datetime(2026, 6, 11, 15, 50, tzinfo=NY))
        assert not blocked and "window" in why
        # weekend -> blocked
        blocked, _ = rm.can_enter(datetime(2026, 6, 13, 10, 0, tzinfo=NY))
        assert not blocked
