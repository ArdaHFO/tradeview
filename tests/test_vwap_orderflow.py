from datetime import datetime, timedelta

from scanner.features.orderflow import (OrderFlowTracker, cvd_bearish_divergence,
                                        cvd_bullish_divergence)
from scanner.features.vwap import SessionVWAP
from scanner.models import Bar, Quote, TradeTick

T0 = datetime(2026, 6, 11, 13, 30)


def test_vwap_simple():
    v = SessionVWAP()
    v.update_trade(TradeTick(ts=T0, price=10.0, size=100))
    v.update_trade(TradeTick(ts=T0, price=20.0, size=100))
    assert abs(v.value - 15.0) < 1e-9
    # volume-weighted: heavier at 10
    v2 = SessionVWAP()
    v2.update_trade(TradeTick(ts=T0, price=10.0, size=300))
    v2.update_trade(TradeTick(ts=T0, price=20.0, size=100))
    assert abs(v2.value - 12.5) < 1e-9


def test_vwap_bar_typical_price():
    v = SessionVWAP()
    v.update_bar(Bar(ts=T0, open=9, high=12, low=9, close=9, volume=100))
    assert abs(v.value - 10.0) < 1e-9   # (12+9+9)/3


def test_vwap_zero_volume_ignored():
    v = SessionVWAP()
    v.update_trade(TradeTick(ts=T0, price=10.0, size=0))
    assert v.value is None


def test_quote_rule_classification():
    f = OrderFlowTracker()
    f.on_quote(Quote(ts=T0, bid=9.99, ask=10.01))
    assert f.on_trade(TradeTick(ts=T0, price=10.01, size=100)) == 1   # at ask = buy
    assert f.on_trade(TradeTick(ts=T0, price=9.99, size=50)) == -1    # at bid = sell
    assert f.cvd == 50.0
    assert f.buy_volume == 100.0 and f.sell_volume == 50.0


def test_tick_rule_fallback():
    f = OrderFlowTracker()   # no quotes at all
    f.on_trade(TradeTick(ts=T0, price=10.00, size=100))   # first: default +1
    assert f.on_trade(TradeTick(ts=T0, price=10.05, size=100)) == 1   # uptick
    assert f.on_trade(TradeTick(ts=T0, price=10.01, size=100)) == -1  # downtick
    assert f.on_trade(TradeTick(ts=T0, price=10.01, size=100)) == -1  # unchanged


def test_taker_imbalance_bounds():
    f = OrderFlowTracker()
    assert f.taker_imbalance() == 0.0
    f.on_quote(Quote(ts=T0, bid=9.99, ask=10.01))
    f.on_trade(TradeTick(ts=T0, price=10.01, size=300))
    f.on_trade(TradeTick(ts=T0, price=9.99, size=100))
    assert abs(f.taker_imbalance() - 0.5) < 1e-9   # (300-100)/400


def test_tape_speed():
    f = OrderFlowTracker(tape_window_sec=60)
    for i in range(30):
        f.on_trade(TradeTick(ts=T0 + timedelta(seconds=i), price=10.0, size=10))
    speed = f.tape_speed(T0 + timedelta(seconds=30))
    assert abs(speed - 0.5) < 1e-9   # 30 trades / 60s window


def test_large_print_detection():
    f = OrderFlowTracker(large_print_multiple=10.0)
    f.on_quote(Quote(ts=T0, bid=9.99, ask=10.01))
    for i in range(60):
        f.on_trade(TradeTick(ts=T0, price=10.01, size=100))
    f.on_trade(TradeTick(ts=T0, price=10.01, size=5000))   # 50x avg
    assert len(f.large_prints) == 1
    assert f.recent_large_print_bias() == 1.0


def test_cvd_divergence_helpers():
    prices = [10 + 0.1 * i for i in range(20)]      # new highs
    cvd_fading = [1000.0 - 10 * i for i in range(20)]
    assert cvd_bearish_divergence(prices, cvd_fading, lookback=20)
    lows = [10 - 0.1 * i for i in range(20)]        # new lows
    cvd_rising = [0.0 + 10 * i for i in range(20)]
    assert cvd_bullish_divergence(lows, cvd_rising, lookback=20)
    assert not cvd_bearish_divergence(prices, [float(i) for i in range(20)], 20)
