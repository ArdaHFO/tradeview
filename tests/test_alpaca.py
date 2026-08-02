"""Tests for the Alpaca data adapter, against a stubbed HTTP session.

No network and no API key: the point is to pin down the parsing and paging
details that are easy to get silently wrong -- nanosecond timestamps, page
tokens, and the quote/trade interleaving that decides whether order flow is
classified by the quote rule or falls back to the tick rule.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import requests

from scanner.data.alpaca import (AlpacaData, AlpacaError, _parse_ts,
                                 feed_real_flow)
from scanner.models import Quote, TradeTick


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Serves a queued list of responses and records the requests made."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params or {}))
        return self._responses.pop(0)


@pytest.fixture
def api():
    def _make(responses):
        a = AlpacaData("key", "secret")
        a.http = FakeSession(responses)
        return a
    return _make


# --- timestamps -----------------------------------------------------------

def test_parse_ts_truncates_nanoseconds():
    """fromisoformat rejects 9 fractional digits; Alpaca sends them."""
    ts = _parse_ts("2026-06-01T13:30:00.123456789Z")
    assert ts == datetime(2026, 6, 1, 13, 30, 0, 123456, tzinfo=timezone.utc)


def test_parse_ts_handles_whole_seconds_and_is_utc():
    ts = _parse_ts("2026-06-01T13:30:00Z")
    assert ts == datetime(2026, 6, 1, 13, 30, tzinfo=timezone.utc)
    assert ts.tzinfo is timezone.utc


# --- credentials ----------------------------------------------------------

def test_missing_credentials_raise():
    with pytest.raises(AlpacaError):
        AlpacaData("", "secret")


def test_credentials_go_into_headers():
    a = AlpacaData("kid", "sec")
    assert a.http.headers["APCA-API-KEY-ID"] == "kid"
    assert a.http.headers["APCA-API-SECRET-KEY"] == "sec"


def test_forbidden_feed_raises_alpaca_error(api):
    a = api([FakeResponse({"message": "subscription required"}, status=403)])
    with pytest.raises(AlpacaError, match="feed"):
        a.trades("AAPL", date(2026, 6, 1))


# --- parsing --------------------------------------------------------------

def test_minute_bars_parsed(api):
    a = api([FakeResponse({"bars": [
        {"t": "2026-06-01T13:30:00Z", "o": 10.0, "h": 10.5, "l": 9.8,
         "c": 10.2, "v": 1500},
    ], "next_page_token": None})])
    bars = a.minute_bars("AAPL", date(2026, 6, 1))
    assert len(bars) == 1
    assert (bars[0].open, bars[0].high, bars[0].low, bars[0].close) == \
        (10.0, 10.5, 9.8, 10.2)
    assert bars[0].volume == 1500


def test_trades_parsed(api):
    a = api([FakeResponse({"trades": [
        {"t": "2026-06-01T13:30:00.5Z", "p": 10.11, "s": 100},
        {"t": "2026-06-01T13:30:01Z", "p": 10.12, "s": 250},
    ], "next_page_token": None})])
    trades = a.trades("AAPL", date(2026, 6, 1))
    assert [t.size for t in trades] == [100, 250]
    assert trades[0].price == pytest.approx(10.11)


def test_quotes_drop_rows_with_no_two_sided_market(api):
    a = api([FakeResponse({"quotes": [
        {"t": "2026-06-01T13:30:00Z", "bp": 10.0, "ap": 10.05},
        {"t": "2026-06-01T13:30:01Z", "bp": 0.0, "ap": 10.06},   # no bid
        {"t": "2026-06-01T13:30:02Z", "bp": 10.01, "ap": 0.0},   # no ask
    ], "next_page_token": None})])
    quotes = a.quotes("AAPL", date(2026, 6, 1))
    assert len(quotes) == 1
    assert (quotes[0].bid, quotes[0].ask) == (10.0, 10.05)


# --- paging ---------------------------------------------------------------

def test_paging_follows_next_page_token(api):
    a = api([
        FakeResponse({"trades": [{"t": "2026-06-01T13:30:00Z", "p": 1.0, "s": 1}],
                      "next_page_token": "tok1"}),
        FakeResponse({"trades": [{"t": "2026-06-01T13:30:01Z", "p": 2.0, "s": 2}],
                      "next_page_token": None}),
    ])
    trades = a.trades("AAPL", date(2026, 6, 1))
    assert len(trades) == 2
    assert a.http.calls[0][1].get("page_token") is None
    assert a.http.calls[1][1]["page_token"] == "tok1"


def test_sip_window_is_clamped_to_the_entitlement_lag(api):
    """A free-plan SIP request must not ask for the most recent 15 minutes."""
    today = datetime.now(timezone.utc).date()
    a = api([FakeResponse({"trades": [], "next_page_token": None})])
    a.feed = "sip"
    a.trades("AAPL", today)
    end = _parse_ts(a.http.calls[0][1]["end"])
    assert end <= datetime.now(timezone.utc)


def test_iex_window_is_not_clamped(api):
    a = api([FakeResponse({"trades": [], "next_page_token": None})])
    a.trades("AAPL", date(2026, 6, 1))
    params = a.http.calls[0][1]
    assert params["feed"] == "iex"
    assert params["end"].startswith("2026-06-02")     # through the post-market


# --- engine bridge --------------------------------------------------------

class RecordingEngine:
    def __init__(self):
        self.events: list[tuple[str, object]] = []

    def on_quote(self, symbol, quote):
        self.events.append(("quote", quote))

    def on_trade(self, symbol, trade):
        self.events.append(("trade", trade))


def _t(sec: int, price: float = 10.0) -> TradeTick:
    return TradeTick(ts=datetime(2026, 6, 1, 13, 30, sec, tzinfo=timezone.utc),
                     price=price, size=100)


def _q(sec: int) -> Quote:
    return Quote(ts=datetime(2026, 6, 1, 13, 30, sec, tzinfo=timezone.utc),
                 bid=9.99, ask=10.01)


def test_feed_real_flow_puts_each_quote_before_the_trade_it_precedes():
    """Quote-rule classification only works if the quote lands first."""
    eng = RecordingEngine()
    n = feed_real_flow(eng, "AAPL", [_t(1), _t(3)], [_q(0), _q(2)],
                       until=_t(9).ts)
    assert n == 2
    assert [kind for kind, _ in eng.events] == ["quote", "trade", "quote", "trade"]


def test_feed_real_flow_stops_at_the_cutoff():
    eng = RecordingEngine()
    n = feed_real_flow(eng, "AAPL", [_t(1), _t(5), _t(9)], [], until=_t(5).ts)
    assert n == 2                                    # the 09s print is held back
    assert len(eng.events) == 2


def test_feed_real_flow_with_no_quotes_still_feeds_trades():
    eng = RecordingEngine()
    n = feed_real_flow(eng, "AAPL", [_t(1), _t(2)], [], until=_t(9).ts)
    assert n == 2
    assert all(kind == "trade" for kind, _ in eng.events)
