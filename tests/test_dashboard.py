"""Tests for the dashboard's backtest-validation panel.

The panel reads whatever `*_trades.json` is newest in the working directory, so
these run inside a tmp_path chdir to stay independent of the repo's real logs.
"""
from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from scanner.config import Config
from scanner.dashboard import create_app


@pytest.fixture
def client_in(tmp_path, monkeypatch):
    """Dashboard client rooted at an empty tmp dir (no scan, no cache)."""
    def _make() -> TestClient:
        monkeypatch.chdir(tmp_path)
        cfg = Config()
        cfg.polygon_api_key = ""          # keep the background scan from doing work
        return TestClient(create_app(cfg))
    return _make


def _write_trades(path, rows) -> None:
    path.write_text(json.dumps(rows), encoding="utf-8")


def _row(r: float, pnl: float, day: str = "2026-06-01") -> dict:
    return {"day": day, "time": "15:00", "symbol": "TEST",
            "setup": "VWAP_REVERSION", "side": "LONG", "entry": 10.0,
            "exit": 11.0, "exit_reason": "target", "r_multiple": r, "pnl": pnl}


def test_validation_reports_unavailable_without_a_trades_file(client_in):
    c = client_in()
    body = c.get("/api/validation").json()
    assert body["available"] is False
    assert "hint" in body


def test_validation_summarises_a_saved_run(tmp_path, client_in):
    _write_trades(tmp_path / "bt_trades.json",
                  [_row(2.0, 100.0) for _ in range(60)]
                  + [_row(-1.0, -50.0) for _ in range(20)])
    body = client_in().get("/api/validation").json()
    assert body["available"] is True
    assert body["n_trades"] == 80
    assert body["passed"] is True                  # 2R at 75% win is a real edge
    assert body["equity"][-1] == pytest.approx(5000.0)
    assert body["profit_factor"]["lo"] > 1.0


def test_validation_flags_a_coin_flip_as_unproven(tmp_path, client_in):
    rows = []
    for i in range(120):                           # alternating +1R / -1R
        rows.append(_row(1.0, 50.0) if i % 2 else _row(-1.0, -50.0))
    _write_trades(tmp_path / "bt_trades.json", rows)
    body = client_in().get("/api/validation").json()
    assert body["available"] is True
    assert body["passed"] is False
    assert body["expectancy"]["lo"] < 0 < body["expectancy"]["hi"]
    assert body["notes"]


def test_validation_picks_the_newest_trades_file(tmp_path, client_in):
    old = tmp_path / "old_trades.json"
    new = tmp_path / "new_trades.json"
    _write_trades(old, [_row(1.0, 50.0) for _ in range(10)])
    _write_trades(new, [_row(-1.0, -50.0) for _ in range(10)])
    os.utime(old, (1_600_000_000, 1_600_000_000))   # force old to be older
    os.utime(new, (1_700_000_000, 1_700_000_000))
    body = client_in().get("/api/validation").json()
    assert body["source"] == "new_trades.json"
    assert body["total_pnl"] == pytest.approx(-500.0)


def test_validation_survives_a_corrupt_trades_file(tmp_path, client_in):
    (tmp_path / "bad_trades.json").write_text("{not json", encoding="utf-8")
    body = client_in().get("/api/validation").json()
    assert body["available"] is False               # degrades, does not 500
    assert "hata" in body["hint"]


def test_index_page_renders_the_validation_section(client_in):
    page = client_in().get("/")
    assert page.status_code == 200
    assert "Backtest Doğrulama" in page.text
