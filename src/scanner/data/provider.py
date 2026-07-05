"""Polygon.io REST adapter for stage-1 screening and reference data.

Live streaming (T.* trades / Q.* quotes / AM.* minute aggregates over
wss://socket.polygon.io/stocks) plugs into the same SymbolContext callbacks;
see `stream_watchlist` for the wiring. Requires a real-time Polygon plan.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from ..models import SymbolSnapshot

log = logging.getLogger(__name__)

BASE = "https://api.polygon.io"


class PolygonProvider:
    def __init__(self, api_key: str, session: requests.Session | None = None) -> None:
        if not api_key:
            raise ValueError("POLYGON_API_KEY is required for live data")
        self.api_key = api_key
        self.http = session or requests.Session()

    def _get(self, path: str, **params) -> dict:
        params["apiKey"] = self.api_key
        for attempt in range(3):
            resp = self.http.get(f"{BASE}{path}", params=params, timeout=30)
            if resp.status_code == 429:           # rate limited: back off and retry
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    # ---- stage-1 inputs -------------------------------------------------

    def full_market_snapshot(self) -> list[SymbolSnapshot]:
        """One call returns every US stock with today's running stats + prevDay.

        avg_daily_volume is approximated with prevDay volume here; for a better
        baseline call `avg_volume_20d` on the shortlist only (cheap).
        """
        data = self._get("/v2/snapshot/locale/us/markets/stocks/tickers")
        out: list[SymbolSnapshot] = []
        for t in data.get("tickers", []):
            day = t.get("day") or {}
            prev = t.get("prevDay") or {}
            last = t.get("lastTrade") or {}
            price = last.get("p") or day.get("c") or 0.0
            prev_close = prev.get("c") or 0.0
            if price <= 0 or prev_close <= 0:
                continue
            out.append(SymbolSnapshot(
                symbol=t.get("ticker", ""),
                price=float(price),
                prev_close=float(prev_close),
                day_volume=float(day.get("v") or 0.0),
                avg_daily_volume=float(prev.get("v") or 0.0),
                day_high=float(day.get("h") or price),
                day_low=float(day.get("l") or price),
            ))
        log.info("snapshot: %d symbols", len(out))
        return out

    def avg_volume_20d(self, symbol: str) -> float | None:
        """Trailing 20-session average daily volume (shortlist refinement)."""
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=35)   # ~20 sessions + weekends/holidays
        data = self._get(
            f"/v2/aggs/ticker/{symbol}/range/1/day/"
            f"{start.isoformat()}/{today.isoformat()}",
            adjusted="true", sort="desc", limit=21)
        results = data.get("results") or []
        vols = [r["v"] for r in results[1:21]]    # skip today's partial bar
        if not vols:
            return None
        return sum(vols) / len(vols)

    def float_shares(self, symbol: str) -> float | None:
        """Free float approximation via weighted shares outstanding."""
        data = self._get(f"/v3/reference/tickers/{symbol}")
        res = data.get("results") or {}
        ws = res.get("weighted_shares_outstanding") or res.get("share_class_shares_outstanding")
        return float(ws) if ws else None

    def enrich_shortlist(self, snapshots: list[SymbolSnapshot]) -> None:
        """Fill float + better avg volume for a SHORT list only (API-cost aware)."""
        for s in snapshots:
            try:
                s.float_shares = self.float_shares(s.symbol)
                avg = self.avg_volume_20d(s.symbol)
                if avg:
                    s.avg_daily_volume = avg
            except requests.RequestException as exc:
                log.warning("enrich failed for %s: %s", s.symbol, exc)

    # ---- live streaming (stage-2) ----------------------------------------

    def stream_watchlist(self, symbols: list[str], on_trade, on_quote, on_bar) -> None:
        """Blocking websocket loop. Callbacks: (symbol, TradeTick/Quote/Bar)."""
        import json

        from websocket import WebSocketApp  # websocket-client

        from ..models import Bar, Quote, TradeTick

        subs = ",".join(f"T.{s},Q.{s},AM.{s}" for s in symbols)

        def _on_open(ws) -> None:
            ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            ws.send(json.dumps({"action": "subscribe", "params": subs}))

        def _on_message(ws, message: str) -> None:
            for ev in json.loads(message):
                kind = ev.get("ev")
                sym = ev.get("sym", "")
                if kind == "T":
                    on_trade(sym, TradeTick(
                        ts=_ms(ev["t"]), price=float(ev["p"]), size=float(ev["s"])))
                elif kind == "Q":
                    on_quote(sym, Quote(
                        ts=_ms(ev["t"]), bid=float(ev["bp"]), ask=float(ev["ap"])))
                elif kind == "AM":
                    on_bar(sym, Bar(
                        ts=_ms(ev["s"]), open=float(ev["o"]), high=float(ev["h"]),
                        low=float(ev["l"]), close=float(ev["c"]), volume=float(ev["v"])))

        ws = WebSocketApp("wss://socket.polygon.io/stocks",
                          on_open=_on_open, on_message=_on_message)
        ws.run_forever(ping_interval=30)


def _ms(epoch_ms: int) -> datetime:
    return datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
