"""End-to-end: synthetic gap-up session must produce a scored, recorded signal."""
from scanner.config import Config
from scanner.demo import run_demo
from scanner.models import SetupType, Side


def test_demo_pipeline_produces_gap_and_go():
    cfg = Config()
    watchlist, signals = run_demo(cfg)

    # Stage 1: only the hot gapper survives
    symbols = [w.symbol for w in watchlist]
    assert symbols == ["HOTX"]
    assert watchlist[0].rvol > 2.0
    assert watchlist[0].gap_pct > 4.0

    # Stage 2: the scripted breakout fires at least one long gap-and-go
    assert signals, "expected at least one signal from the scripted session"
    sig = signals[0]
    assert sig.symbol == "HOTX"
    assert sig.setup == SetupType.GAP_AND_GO
    assert sig.side == Side.LONG
    assert sig.confluence >= cfg.signal.threshold
    assert sig.stop < sig.entry < sig.target
    assert sig.position_size > 0
    assert sig.risk_dollars <= cfg.risk.equity * cfg.risk.risk_per_trade_pct / 100 + 1e-6
    assert sig.reasons
