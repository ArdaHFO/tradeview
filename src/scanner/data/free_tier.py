"""Free-tier Polygon scanner: grouped daily bars (EOD) -> watchlist.

The free plan blocks the live snapshot endpoint but allows grouped daily
aggregates for completed sessions at 5 requests/minute. This module builds
SymbolSnapshots from the last N completed sessions:
  price       = latest session close
  prev_close  = prior session close      (gap_pct == day change %)
  day_volume  = latest session volume
  avg_volume  = mean volume of the prior sessions
Float lookup is skipped (1 request per symbol is too slow at 5/min).
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone

import requests

from ..models import SymbolSnapshot

log = logging.getLogger(__name__)

BASE = "https://api.polygon.io"
THROTTLE_SEC = 13          # free tier: 5 requests/minute


class FreeTierScanner:
    def __init__(self, api_key: str, lookback_days: int = 6) -> None:
        self.api_key = api_key
        self.lookback_days = lookback_days
        self.http = requests.Session()
        self._last_call = 0.0
        self._cache: dict[date, dict[str, dict] | None] = {}   # completed days only

    def _grouped(self, day: date) -> dict[str, dict] | None:
        """symbol -> {o,h,l,c,v} for one completed session; None if unavailable.
        Cached in memory: multi-day backtests reuse overlapping sessions."""
        if day in self._cache:
            return self._cache[day]
        for attempt in range(4):
            wait = THROTTLE_SEC - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.http.get(
                f"{BASE}/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}",
                params={"apiKey": self.api_key, "adjusted": "true"}, timeout=60)
            if resp.status_code == 429:     # shared rate limit: back off, retry
                time.sleep(15 * (attempt + 1))
                continue
            if resp.status_code in (403, 404):
                self._cache[day] = None     # not entitled (too recent) or holiday
                return None
            resp.raise_for_status()
            results = resp.json().get("results") or []
            data = {r["T"]: r for r in results} if results else None
            self._cache[day] = data
            return data
        resp.raise_for_status()
        return None

    def scan(self, progress=None,
             end_date: date | None = None) -> tuple[date, list[SymbolSnapshot]]:
        """Fetch the last `lookback_days` completed sessions and build snapshots.

        `end_date`: walk back starting from this date (inclusive); defaults to
        today. Pass trading_day - 1 to build a bias-free watchlist for a
        backtest of trading_day.

        Returns (as_of_date, snapshots). Takes ~13s per session fetched
        (free-tier throttle); call from a background thread in the UI.
        """
        days: list[tuple[date, dict[str, dict]]] = []
        d = end_date or datetime.now(timezone.utc).date()
        attempts = 0
        while len(days) < self.lookback_days and attempts < 20:
            attempts += 1
            if d.weekday() >= 5:             # skip weekends
                d -= timedelta(days=1)
                continue
            if progress:
                progress(f"fetching {d.isoformat()} ({len(days)+1}/{self.lookback_days})")
            data = self._grouped(d)
            if data:
                days.append((d, data))
                log.info("grouped %s: %d symbols", d, len(data))
            d -= timedelta(days=1)
        if len(days) < 2:
            raise RuntimeError("need at least 2 completed sessions from Polygon")

        days.sort(key=lambda x: x[0])        # oldest -> newest
        as_of, latest = days[-1]
        _, prior_day = days[-2]
        history = days[:-1]                  # volume baseline

        snapshots: list[SymbolSnapshot] = []
        for sym, bar in latest.items():
            prev = prior_day.get(sym)
            if not prev:
                continue
            vols = [data[sym]["v"] for _, data in history if sym in data]
            if not vols:
                continue
            snapshots.append(SymbolSnapshot(
                symbol=sym,
                price=float(bar["c"]),
                prev_close=float(prev["c"]),
                day_volume=float(bar["v"]),
                avg_daily_volume=sum(vols) / len(vols),
                day_high=float(bar["h"]),
                day_low=float(bar["l"]),
            ))
        return as_of, snapshots
