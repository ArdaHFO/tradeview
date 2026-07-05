"""Session-anchored VWAP, incrementally updated from 1-minute bars or raw trades."""
from __future__ import annotations

from ..models import Bar, TradeTick


class SessionVWAP:
    """Cumulative VWAP anchored at the session open.

    Feed it either bars (uses typical price (H+L+C)/3) or raw trades
    (exact). Call `reset()` at each new session.
    """

    def __init__(self) -> None:
        self._pv = 0.0       # cumulative price * volume
        self._vol = 0.0      # cumulative volume
        self._pv2 = 0.0      # cumulative price^2 * volume (for band stdev)

    def reset(self) -> None:
        self._pv = 0.0
        self._vol = 0.0
        self._pv2 = 0.0

    def update_bar(self, bar: Bar) -> None:
        tp = (bar.high + bar.low + bar.close) / 3.0
        self._add(tp, bar.volume)

    def update_trade(self, trade: TradeTick) -> None:
        self._add(trade.price, trade.size)

    def _add(self, price: float, volume: float) -> None:
        if volume <= 0:
            return
        self._pv += price * volume
        self._vol += volume
        self._pv2 += price * price * volume

    @property
    def value(self) -> float | None:
        if self._vol <= 0:
            return None
        return self._pv / self._vol

    @property
    def stdev(self) -> float | None:
        """Volume-weighted standard deviation of price around VWAP."""
        if self._vol <= 0:
            return None
        mean = self._pv / self._vol
        var = self._pv2 / self._vol - mean * mean
        return var ** 0.5 if var > 0 else 0.0

    def band(self, n_std: float) -> tuple[float, float] | None:
        v, s = self.value, self.stdev
        if v is None or s is None:
            return None
        return v - n_std * s, v + n_std * s

    def distance_in_std(self, price: float) -> float | None:
        """Signed distance of `price` from VWAP in stdev units (+ above, - below)."""
        v, s = self.value, self.stdev
        if v is None or s is None or s == 0:
            return None
        return (price - v) / s
