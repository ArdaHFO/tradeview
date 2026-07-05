"""Web dashboard: watchlist + signals in the browser (FastAPI, single page).

Run:  python main.py dashboard   ->  http://localhost:8000
Free-tier mode: EOD scan over completed sessions, cached to watchlist_cache.json.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Config
from .data.free_tier import FreeTierScanner
from .levels import playbook_levels
from .stage1_screener import screen

log = logging.getLogger(__name__)

CACHE_FILE = Path("watchlist_cache.json")


class DashboardState:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.status = "idle"
        self.progress = ""
        self.as_of: str | None = None
        self.scanned_at: str | None = None
        self.watchlist: list[dict] = []
        self.error: str | None = None
        self._load_cache()

    # ---- cache ----------------------------------------------------------

    def _load_cache(self) -> None:
        if CACHE_FILE.exists():
            try:
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                self.as_of = data.get("as_of")
                self.scanned_at = data.get("scanned_at")
                self.watchlist = data.get("watchlist", [])
                self.status = "ready (cached)"
            except (json.JSONDecodeError, OSError):
                pass

    def _save_cache(self) -> None:
        CACHE_FILE.write_text(json.dumps({
            "as_of": self.as_of,
            "scanned_at": self.scanned_at,
            "watchlist": self.watchlist,
        }), encoding="utf-8")

    # ---- scanning -------------------------------------------------------

    def start_scan(self) -> bool:
        with self.lock:
            if self.status == "scanning":
                return False
            self.status = "scanning"
            self.error = None
        threading.Thread(target=self._scan, daemon=True).start()
        return True

    def _scan(self) -> None:
        try:
            scanner = FreeTierScanner(self.cfg.polygon_api_key)
            as_of, snapshots = scanner.scan(progress=self._set_progress)
            items = screen(snapshots, self.cfg.screener, elapsed_fraction=1.0)
            snap_by_sym = {s.symbol: s for s in snapshots}
            rows = []
            for w in items:
                lv = playbook_levels(snap_by_sym[w.symbol])
                rows.append({
                    "symbol": w.symbol,
                    "rvol": round(w.rvol, 1),
                    "gap_pct": round(w.gap_pct, 1),
                    "price": round(w.price, 2),
                    "range_pct": round(w.day_range_pct, 1),
                    "heat": round(w.score),
                    "side": lv.side if lv else "—",
                    "entry": lv.entry if lv else None,
                    "stop": lv.stop if lv else None,
                    "target": lv.target if lv else None,
                })
            with self.lock:
                self.as_of = as_of.isoformat()
                self.scanned_at = datetime.now(timezone.utc).isoformat()
                self.watchlist = rows
                self.status = "ready"
                self.progress = ""
            self._save_cache()
        except Exception as exc:                # surface any failure in the UI
            log.exception("scan failed")
            with self.lock:
                self.status = "error"
                self.error = str(exc)

    def _set_progress(self, msg: str) -> None:
        with self.lock:
            self.progress = msg

    # ---- signals --------------------------------------------------------

    def signals(self) -> list[dict]:
        db = Path(self.cfg.db_path)
        if not db.exists():
            return []
        conn = sqlite3.connect(str(db))
        try:
            rows = conn.execute(
                "SELECT ts, symbol, setup, side, entry, stop, target, confluence,"
                " position_size, risk_dollars, reasons_json FROM signals"
                " ORDER BY id DESC LIMIT 50").fetchall()
        finally:
            conn.close()
        return [{
            "ts": r[0], "symbol": r[1], "setup": r[2], "side": r[3],
            "entry": r[4], "stop": r[5], "target": r[6], "confluence": r[7],
            "size": r[8], "risk": r[9], "reasons": json.loads(r[10]),
        } for r in rows]


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="US Day-Trading Scanner")
    state = DashboardState(cfg)
    if not state.watchlist:                     # first launch: scan immediately
        state.start_scan()

    @app.get("/api/state")
    def api_state() -> JSONResponse:
        with state.lock:
            return JSONResponse({
                "status": state.status,
                "progress": state.progress,
                "error": state.error,
                "as_of": state.as_of,
                "scanned_at": state.scanned_at,
                "mode": "FREE TIER — EOD data (canlı veri için ücretli plan)",
                "watchlist": state.watchlist,
                "signals": state.signals(),
            })

    @app.post("/api/rescan")
    def api_rescan() -> JSONResponse:
        started = state.start_scan()
        return JSONResponse({"started": started})

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    return app


PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>US Day-Trading Scanner</title>
<style>
 :root { color-scheme: dark; }
 body { background:#0d1117; color:#e6edf3; font:14px/1.5 'Segoe UI',sans-serif;
        margin:0; padding:24px; }
 h1 { font-size:20px; margin:0 0 4px; }
 .sub { color:#8b949e; font-size:12px; margin-bottom:16px; }
 .badge { display:inline-block; padding:2px 10px; border-radius:12px;
          font-size:11px; font-weight:600; margin-left:8px; }
 .warn { background:#3d2e00; color:#e3b341; border:1px solid #e3b341; }
 .ok   { background:#0f2e1b; color:#3fb950; border:1px solid #3fb950; }
 .err  { background:#3d0d0d; color:#f85149; border:1px solid #f85149; }
 button { background:#238636; color:#fff; border:0; border-radius:6px;
          padding:8px 18px; font-weight:600; cursor:pointer; }
 button:disabled { background:#30363d; cursor:wait; }
 table { border-collapse:collapse; width:100%; margin-top:10px; }
 th { text-align:left; color:#8b949e; font-size:11px; text-transform:uppercase;
      border-bottom:1px solid #30363d; padding:6px 10px; }
 td { padding:7px 10px; border-bottom:1px solid #21262d; font-variant-numeric:tabular-nums; }
 tr:hover td { background:#161b22; }
 .pos { color:#3fb950; } .neg { color:#f85149; }
 .heat { font-weight:700; color:#e3b341; }
 .grid { display:grid; grid-template-columns: 1fr 1fr; gap:32px; }
 @media (max-width:1000px){ .grid { grid-template-columns:1fr; } }
 .empty { color:#8b949e; padding:18px 10px; font-style:italic; }
 .long { color:#3fb950; font-weight:700; } .short { color:#f85149; font-weight:700; }
</style></head><body>
<h1>US Day-Trading Scanner <span id="mode" class="badge warn"></span>
    <span id="status" class="badge ok"></span></h1>
<div class="sub">Veri günü: <b id="asof">—</b> · Son tarama: <span id="scanned">—</span>
 · <button id="rescan" onclick="rescan()">Yeniden Tara</button></div>
<div class="grid">
<div>
 <h3>🔥 Watchlist — "In Play" Hisseler <small style="color:#8b949e;font-weight:400">
   (entry/stop/target = plan seviyesi; canlı CVD teyidi olmadan emir değildir)</small></h3>
 <table><thead><tr><th>#</th><th>Sembol</th><th>RVOL</th><th>Değişim %</th>
 <th>Fiyat</th><th>Heat</th><th>Yön</th><th>Entry</th><th>Stop</th><th>Target</th></tr></thead>
 <tbody id="wl"><tr><td colspan="10" class="empty">yükleniyor…</td></tr></tbody></table>
</div>
<div>
 <h3>⚡ Sinyaller (son 50)</h3>
 <table><thead><tr><th>Zaman</th><th>Sembol</th><th>Setup</th><th>Yön</th>
 <th>Entry</th><th>Stop</th><th>Target</th><th>Skor</th></tr></thead>
 <tbody id="sg"><tr><td colspan="8" class="empty">henüz sinyal yok</td></tr></tbody></table>
</div>
</div>
<script>
async function refresh(){
  const r = await fetch('/api/state'); const s = await r.json();
  document.getElementById('mode').textContent = s.mode;
  const st = document.getElementById('status');
  st.textContent = s.status + (s.progress ? ' · ' + s.progress : '');
  st.className = 'badge ' + (s.status==='error' ? 'err' : (s.status==='scanning' ? 'warn' : 'ok'));
  if (s.error) st.textContent += ' — ' + s.error;
  document.getElementById('asof').textContent = s.as_of || '—';
  document.getElementById('scanned').textContent =
      s.scanned_at ? new Date(s.scanned_at).toLocaleTimeString() : '—';
  document.getElementById('rescan').disabled = (s.status === 'scanning');
  const wl = document.getElementById('wl');
  if (s.watchlist.length){
    wl.innerHTML = s.watchlist.map((w,i)=>`<tr><td>${i+1}</td>
      <td><b>${w.symbol}</b></td><td>${w.rvol}x</td>
      <td class="${w.gap_pct>=0?'pos':'neg'}">${w.gap_pct>=0?'+':''}${w.gap_pct}%</td>
      <td>$${w.price}</td><td class="heat">${w.heat}</td>
      <td class="${w.side==='LONG'?'long':'short'}">${w.side}</td>
      <td>${w.entry ?? '—'}</td><td>${w.stop ?? '—'}</td><td>${w.target ?? '—'}</td>
      </tr>`).join('');
  } else if (s.status !== 'scanning') {
    wl.innerHTML = '<tr><td colspan="10" class="empty">filtreyi geçen hisse yok</td></tr>';
  }
  const sg = document.getElementById('sg');
  if (s.signals.length){
    sg.innerHTML = s.signals.map(x=>`<tr title="${x.reasons.join('; ')}">
      <td>${new Date(x.ts).toLocaleTimeString()}</td><td><b>${x.symbol}</b></td>
      <td>${x.setup}</td><td class="${x.side==='LONG'?'long':'short'}">${x.side}</td>
      <td>${x.entry.toFixed(2)}</td><td>${x.stop.toFixed(2)}</td>
      <td>${x.target.toFixed(2)}</td><td>${x.confluence.toFixed(2)}</td></tr>`).join('');
  }
}
async function rescan(){ await fetch('/api/rescan',{method:'POST'}); refresh(); }
refresh(); setInterval(refresh, 5000);
</script></body></html>"""
