"""Order flow features: CVD, taker imbalance, tape speed, large prints.

Aggressor-side classification:
  1. Quote rule (preferred): trade at/above ask = buyer-aggressive,
     at/below bid = seller-aggressive, between = tick rule fallback.
  2. Tick rule: uptick = buy, downtick = sell, unchanged = previous side.

US trade feeds (e.g. Polygon T.*) do not carry the aggressor flag, so this
classification is the standard practical approximation used by footprint tools.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

from ..models import Quote, TradeTick


class OrderFlowTracker:
    """Streaming order-flow state for one symbol. Feed quotes then trades."""

    def __init__(self, tape_window_sec: int = 60,
                 large_print_multiple: float = 10.0,
                 max_tape: int = 5000) -> None:
        self.cvd = 0.0                       # cumulative volume delta (shares)
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self._last_quote: Quote | None = None
        self._last_price: float | None = None
        self._last_side = 1                  # +1 buy / -1 sell (tick-rule memory)
        self._tape: deque[TradeTick] = deque(maxlen=max_tape)
        self._tape_window = timedelta(seconds=tape_window_sec)
        self._large_print_multiple = large_print_multiple
        self._size_sum = 0.0
        self._size_n = 0
        self.large_prints: list[tuple[TradeTick, int]] = []   # (trade, side)
        self._cvd_history: list[tuple[datetime, float]] = []  # minute-close snapshots
        self._cvd_minute: datetime | None = None

    # ---- inputs -------------------------------------------------------

    def on_quote(self, quote: Quote) -> None:
        self._last_quote = quote

    def on_trade(self, trade: TradeTick) -> int:
        """Process a trade; returns classified side (+1 buy, -1 sell)."""
        side = self._classify(trade)
        signed = side * trade.size
        self.cvd += signed
        if side > 0:
            self.buy_volume += trade.size
        else:
            self.sell_volume += trade.size

        self._tape.append(trade)
        self._size_sum += trade.size
        self._size_n += 1
        avg_size = self._size_sum / self._size_n
        if self._size_n >= 50 and trade.size >= self._large_print_multiple * avg_size:
            self.large_prints.append((trade, side))

        self._snapshot_cvd(trade.ts)
        self._last_price = trade.price
        self._last_side = side
        return side

    def _classify(self, trade: TradeTick) -> int:
        q = self._last_quote
        if q is not None and q.bid > 0 and q.ask > q.bid:
            if trade.price >= q.ask:
                return 1
            if trade.price <= q.bid:
                return -1
        # tick rule fallback
        if self._last_price is not None:
            if trade.price > self._last_price:
                return 1
            if trade.price < self._last_price:
                return -1
        return self._last_side

    def _snapshot_cvd(self, ts: datetime) -> None:
        minute = ts.replace(second=0, microsecond=0)
        if self._cvd_minute is None:
            self._cvd_minute = minute
        elif minute > self._cvd_minute:
            self._cvd_history.append((self._cvd_minute, self.cvd))
            self._cvd_minute = minute

    # ---- features -----------------------------------------------------

    def taker_imbalance(self) -> float:
        """(buy - sell) / total in [-1, 1]. 0 when no volume yet."""
        total = self.buy_volume + self.sell_volume
        if total <= 0:
            return 0.0
        return (self.buy_volume - self.sell_volume) / total

    def tape_speed(self, now: datetime) -> float:
        """Trades per second over the trailing window."""
        cutoff = now - self._tape_window
        n = sum(1 for t in self._tape if t.ts >= cutoff)
        return n / self._tape_window.total_seconds()

    def cvd_slope(self, minutes: int = 5) -> float | None:
        """CVD change over the last N completed minutes (shares). None if no history."""
        hist = self._cvd_history
        if len(hist) < 2:
            return None
        window = hist[-minutes:] if len(hist) >= minutes else hist
        return window[-1][1] - window[0][1]

    def cvd_minute_series(self) -> list[float]:
        return [v for _, v in self._cvd_history]

    def recent_large_print_bias(self, last_n: int = 5) -> float:
        """Mean side of the last N large prints in [-1, 1]; 0 if none."""
        if not self.large_prints:
            return 0.0
        sides = [s for _, s in self.large_prints[-last_n:]]
        return sum(sides) / len(sides)


def cvd_bearish_divergence(prices: list[float], cvd_series: list[float],
                           lookback: int = 20) -> bool:
    """Price new high over lookback while CVD fails to make a new high."""
    if len(prices) < lookback or len(cvd_series) < lookback:
        return False
    p = prices[-lookback:]
    c = cvd_series[-lookback:]
    return p[-1] >= max(p) and c[-1] < max(c[:-1])


def cvd_bullish_divergence(prices: list[float], cvd_series: list[float],
                           lookback: int = 20) -> bool:
    """Price new low over lookback while CVD fails to make a new low."""
    if len(prices) < lookback or len(cvd_series) < lookback:
        return False
    p = prices[-lookback:]
    c = cvd_series[-lookback:]
    return p[-1] <= min(p) and c[-1] > min(c[:-1])
