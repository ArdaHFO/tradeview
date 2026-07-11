from forecaster.webapp import _parse_timeframes


def test_splits_valid_multi_timeframe_string():
    assert _parse_timeframes("1d,1wk,1mo") == ["1d", "1wk", "1mo"]


def test_single_timeframe():
    assert _parse_timeframes("1d") == ["1d"]


def test_none_defaults_to_1d():
    assert _parse_timeframes(None) == ["1d"]


def test_garbage_falls_back_to_1d():
    # This is the exact string a stale client used to send as a single
    # literal yfinance `interval=` value, which Yahoo rejected outright.
    assert _parse_timeframes("bogus,also-bogus") == ["1d"]


def test_dedupes_preserving_order():
    assert _parse_timeframes("1d,1d,1wk,1d") == ["1d", "1wk"]
