"""Alpaca market-data adapter: real trade prints instead of synthesised flow.

Why this exists: the Polygon free tier serves no tick history, so the backtester
approximates order flow from bar shape (`buy_ratio = (close-low)/(high-low)`).
That makes CVD a near-restatement of price rather than independent evidence,
which is the prime suspect for why the order-flow-gated setups never validated.
Alpaca serves historical trades and quotes on the free plan, so `OrderFlowTracker`
can be fed the real prints it was designed for.

Feed entitlement is the one thing to get right:
  * `iex`  -- always available, but a single venue at roughly 2.5% of consolidated
              volume. Real prints, biased sample: CVD direction is meaningful,
              absolute volume is not.
  * `sip`  -- the full consolidated tape. Alpaca documents it as subscription
              only, while the free plan is widely reported to serve historical
              SIP as long as `end` is at least 15 minutes in the past. Rather
              than guess, `probe_feeds()` asks the API what this key can do.

Docs: https://docs.alpaca.markets/us/docs/historical-stock-data-1
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, time as dtime, timedelta, timezone

import requests

from ..models import Bar, Quote, TradeTick

log = logging.getLogger(__name__)

DATA_BASE = "https://data.alpaca.markets/v2"
PAGE_LIMIT = 10_000
SIP_LAG = timedelta(minutes=15)     # free plan may serve SIP only this far back
MAX_RETRIES = 4


class AlpacaError(RuntimeError):
    """Raised for auth/entitlement failures that retrying will not fix."""


def _parse_ts(raw: str) -> datetime:
    """Parse Alpaca's RFC-3339 timestamps, which carry nanosecond precision.

    `datetime.fromisoformat` rejects 9 fractional digits, so the fraction is
    truncated to microseconds before parsing.
    """
    txt = raw.rstrip("Z")
    if "." in txt:
        head, frac = txt.split(".", 1)
        txt = f"{head}.{frac[:6]}"
    return datetime.fromisoformat(txt).replace(tzinfo=timezone.utc)


def _session_bounds(day: date) -> tuple[datetime, datetime]:
    """UTC window covering a full US trading day, pre/post market included.

    Deliberately wider than RTH: the engine filters to RTH itself, and the extra
    prints give the tick-rule classifier price context before the open.
    """
    start = datetime.combine(day, dtime(8, 0), tzinfo=timezone.utc)   # 04:00 ET
    end = datetime.combine(day, dtime(1, 0), tzinfo=timezone.utc) + timedelta(days=1)
    return start, end


class AlpacaData:
    """Historical bars / trades / quotes for one API key."""

    def __init__(self, key_id: str, secret_key: str, feed: str = "iex") -> None:
        if not key_id or not secret_key:
            raise AlpacaError("Alpaca anahtarları eksik "
                              "(ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        self.feed = feed
        self.http = requests.Session()
        self.http.headers.update({"APCA-API-KEY-ID": key_id,
                                  "APCA-API-SECRET-KEY": secret_key})

    # ---- transport ------------------------------------------------------

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(MAX_RETRIES):
            resp = self.http.get(f"{DATA_BASE}{path}", params=params, timeout=60)
            if resp.status_code == 429:              # rate limited: back off
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code in (401, 403):
                raise AlpacaError(
                    f"{resp.status_code} — anahtar geçersiz ya da '{params.get('feed')}' "
                    f"feed'i bu plana kapalı: {resp.text[:200]}")
            resp.raise_for_status()
            return resp.json()
        raise AlpacaError(f"{path}: {MAX_RETRIES} denemede rate limit aşılamadı")

    def _paged(self, path: str, params: dict, key: str) -> list[dict]:
        """Follow Alpaca's next_page_token until the window is exhausted."""
        out: list[dict] = []
        token: str | None = None
        while True:
            page = dict(params, limit=PAGE_LIMIT)
            if token:
                page["page_token"] = token
            body = self._get(path, page)
            out.extend(body.get(key) or [])
            token = body.get("next_page_token")
            if not token:
                return out

    def _window(self, day: date) -> dict:
        """Query window, clamped so a free-plan SIP request stays entitled."""
        start, end = _session_bounds(day)
        if self.feed == "sip":
            end = min(end, datetime.now(timezone.utc) - SIP_LAG)
        return {"start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "feed": self.feed}

    # ---- data -----------------------------------------------------------

    def minute_bars(self, symbol: str, day: date) -> list[Bar]:
        rows = self._paged(f"/stocks/{symbol}/bars",
                           dict(self._window(day), timeframe="1Min"), "bars")
        return [Bar(ts=_parse_ts(r["t"]), open=float(r["o"]), high=float(r["h"]),
                    low=float(r["l"]), close=float(r["c"]),
                    volume=float(r["v"])) for r in rows]

    def trades(self, symbol: str, day: date) -> list[TradeTick]:
        rows = self._paged(f"/stocks/{symbol}/trades", self._window(day), "trades")
        return [TradeTick(ts=_parse_ts(r["t"]), price=float(r["p"]),
                          size=float(r["s"])) for r in rows]

    def quotes(self, symbol: str, day: date) -> list[Quote]:
        """NBBO quotes. Optional: they upgrade aggressor classification from the
        tick rule to the quote rule, at the cost of a much larger download."""
        rows = self._paged(f"/stocks/{symbol}/quotes", self._window(day), "quotes")
        return [Quote(ts=_parse_ts(r["t"]), bid=float(r["bp"]), ask=float(r["ap"]))
                for r in rows if float(r["bp"]) > 0 and float(r["ap"]) > 0]

    # ---- entitlement ----------------------------------------------------

    def probe_feeds(self, symbol: str = "AAPL") -> dict[str, str]:
        """Ask the API which feeds this key can actually read historical trades on.

        The docs and community reports disagree about free-plan SIP access, and
        the answer decides whether backtests see the full tape or 2.5% of it.
        """
        day = datetime.now(timezone.utc).date() - timedelta(days=3)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
        window = {"start": datetime.combine(day, dtime(14, 30),
                                            tzinfo=timezone.utc).isoformat(),
                  "end": datetime.combine(day, dtime(14, 31),
                                          tzinfo=timezone.utc).isoformat()}
        out: dict[str, str] = {}
        for feed in ("iex", "sip"):
            try:
                body = self._get(f"/stocks/{symbol}/trades",
                                 dict(window, feed=feed, limit=10))
                n = len(body.get("trades") or [])
                out[feed] = f"OK — {n} print döndü" if n else "OK ama boş döndü"
            except AlpacaError as exc:
                out[feed] = f"KAPALI — {exc}"
            except requests.RequestException as exc:
                out[feed] = f"HATA — {exc}"
        return out


# ---- engine bridge ------------------------------------------------------

def feed_real_flow(engine, symbol: str, trades: list[TradeTick],
                   quotes: list[Quote], until: datetime) -> int:
    """Push every print (and preceding quote) up to `until` into the engine.

    Quotes are merged ahead of the trades they precede so the tracker classifies
    with the quote rule rather than falling back to the tick rule. Returns the
    number of trades consumed so the caller can advance its cursor.

    Both inputs must be sorted by timestamp.
    """
    qi = 0
    consumed = 0
    for trade in trades:
        if trade.ts > until:
            break
        while qi < len(quotes) and quotes[qi].ts <= trade.ts:
            engine.on_quote(symbol, quotes[qi])
            qi += 1
        engine.on_trade(symbol, trade)
        consumed += 1
    return consumed
