"""Telegram alerts. Silently no-ops if credentials are missing (console still logs)."""
from __future__ import annotations

import logging

import requests

from ..models import Signal

log = logging.getLogger(__name__)


def format_signal(sig: Signal) -> str:
    fam = sig.scores.as_dict()
    fam_str = " | ".join(f"{k[:4]}:{v:.2f}" for k, v in fam.items())
    arrow = "🟢 LONG" if sig.side.value == "LONG" else "🔴 SHORT"
    lines = [
        f"{arrow} {sig.symbol} — {sig.setup.value}",
        f"score {sig.confluence:.2f}  [{fam_str}]",
        f"entry {sig.entry:.2f} | stop {sig.stop:.2f} | target {sig.target:.2f}",
        f"size {sig.position_size} sh | risk ${sig.risk_dollars:.0f}",
        "why: " + "; ".join(sig.reasons),
    ]
    return "\n".join(lines)


class TelegramAlerter:
    def __init__(self, bot_token: str, chat_id: str) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def send(self, sig: Signal) -> bool:
        text = format_signal(sig)
        log.info("SIGNAL\n%s", text)
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=10,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.warning("telegram send failed: %s", exc)
            return False
