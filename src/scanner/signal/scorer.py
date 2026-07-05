"""Confluence scorer: weighted family scores -> accept/reject + final Signal."""
from __future__ import annotations

from datetime import datetime

from ..config import SignalConfig
from ..models import SetupCandidate, SetupType, Side, Signal


class ConfluenceScorer:
    def __init__(self, cfg: SignalConfig) -> None:
        self.cfg = cfg
        total = cfg.w_orderflow + cfg.w_position + cfg.w_structure + cfg.w_volatility
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"family weights must sum to 1.0, got {total}")

    def confluence(self, cand: SetupCandidate) -> float:
        s = cand.scores
        return (self.cfg.w_orderflow * s.orderflow
                + self.cfg.w_position * s.position
                + self.cfg.w_structure * s.structure
                + self.cfg.w_volatility * s.volatility)

    def score(self, cand: SetupCandidate, now: datetime,
              position_size: int, risk_dollars: float) -> Signal | None:
        """Return a Signal if the candidate clears the threshold, else None."""
        total = self.confluence(cand)
        if total < self.cfg.threshold:
            return None
        target = self._target(cand)
        return Signal(
            ts=now,
            symbol=cand.symbol,
            setup=cand.setup,
            side=cand.side,
            entry=cand.entry,
            stop=cand.stop,
            target=target,
            confluence=round(total, 3),
            scores=cand.scores,
            position_size=position_size,
            risk_dollars=risk_dollars,
            reasons=cand.reasons,
        )

    def _target(self, cand: SetupCandidate) -> float:
        """Default 2R target; reversion setups aim at VWAP-ish 1.5R instead."""
        r = abs(cand.entry - cand.stop)
        mult = (1.5 if cand.setup == SetupType.VWAP_REVERSION
                else self.cfg.target_r_multiple)
        if cand.side == Side.LONG:
            return cand.entry + mult * r
        return cand.entry - mult * r
