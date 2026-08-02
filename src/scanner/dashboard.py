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
from .validation import (bootstrap, equity_curve, load_trades, monte_carlo,
                         verdict)

log = logging.getLogger(__name__)

CACHE_FILE = Path("watchlist_cache.json")
# Fewer resamples than the CLI default: the dashboard re-runs this on every
# poll and 2k is already tight enough for the interval to be stable to ~0.01R.
DASH_ITERATIONS = 2_000


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
        self._val_stamp: tuple | None = None
        self._val_cache: dict = {}
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

    # ---- backtest validation --------------------------------------------

    def validation(self) -> dict:
        """Bootstrap/Monte Carlo summary of the newest saved backtest run.

        Cached on the trades file's mtime: the resampling costs ~1s and the
        page polls every few seconds.
        """
        files = sorted(Path(".").glob("*_trades.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            return {"available": False,
                    "hint": "python main.py backtest --days 30 --save-trades bt.json"}
        newest = files[0]
        stamp = (str(newest), newest.stat().st_mtime)
        with self.lock:
            if self._val_stamp == stamp:
                return self._val_cache
        try:
            payload = self._compute_validation(newest)
        except Exception as exc:                # never take the page down
            log.exception("validation failed")
            payload = {"available": False, "hint": f"doğrulama hatası: {exc}"}
        with self.lock:
            self._val_stamp, self._val_cache = stamp, payload
        return payload

    def _compute_validation(self, path: Path) -> dict:
        trades = load_trades(path)
        if len(trades) < 2:
            return {"available": False, "hint": f"{path.name}: yetersiz işlem"}
        boot = bootstrap(trades, iterations=DASH_ITERATIONS)
        mc = monte_carlo(trades, self.cfg.risk.equity, iterations=DASH_ITERATIONS)
        vd = verdict(boot, mc)
        days = sorted({t.day for t in trades if t.day})
        return {
            "available": True,
            "source": path.name,
            "n_trades": len(trades),
            "n_days": len(days),
            "span": f"{days[0]} → {days[-1]}" if days else "—",
            "setups": sorted({t.setup for t in trades}),
            "expectancy": _iv(boot.expectancy_r),
            "profit_factor": _iv(boot.profit_factor),
            "win_rate": _iv(boot.win_rate),
            "p_no_edge": boot.p_no_edge,
            "trades_needed": boot.trades_needed,
            "total_pnl": mc.realized_pnl,
            "max_dd": mc.realized_max_dd,
            "max_dd_ci": _iv(mc.max_dd),
            "final_pnl_ci": _iv(mc.final_pnl),
            "p_losing_run": mc.p_losing_run,
            "p_deep_dd": mc.p_deep_dd,
            "dd_limit_pct": mc.dd_limit_pct,
            "equity": [round(v, 2) for v in equity_curve([t.pnl for t in trades])],
            "passed": vd.passed,
            "headline": vd.headline,
            "notes": vd.notes,
        }


def _iv(interval) -> dict:
    return {"point": interval.point, "lo": interval.lo, "hi": interval.hi}


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

    @app.get("/api/validation")
    def api_validation() -> JSONResponse:
        return JSONResponse(state.validation())

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
 section { margin-top:34px; border-top:1px solid #21262d; padding-top:18px; }
 .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
          gap:12px; margin:14px 0; }
 .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:12px 14px; }
 .card .lbl { color:#8b949e; font-size:11px; text-transform:uppercase;
              letter-spacing:.4px; }
 .card .val { font-size:22px; font-weight:700; margin:3px 0; font-variant-numeric:tabular-nums; }
 .card .ci { font-size:11px; color:#8b949e; font-variant-numeric:tabular-nums; }
 .verdict { border-radius:8px; padding:14px 16px; margin:14px 0; font-weight:600; }
 .verdict.fail { background:#3d0d0d33; border:1px solid #f85149; color:#f85149; }
 .verdict.pass { background:#0f2e1b33; border:1px solid #3fb950; color:#3fb950; }
 .verdict ul { margin:10px 0 0; padding-left:20px; font-weight:400; color:#e6edf3;
               font-size:13px; }
 .verdict li { margin:4px 0; }
 .cibar { position:relative; height:6px; background:#30363d; border-radius:3px;
          margin-top:7px; }
 .cibar i { position:absolute; height:100%; border-radius:3px; background:#58a6ff; }
 .cibar b { position:absolute; width:2px; height:12px; top:-3px; background:#e6edf3; }
 .cibar u { position:absolute; width:1px; height:12px; top:-3px; background:#8b949e; }
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

<section id="valsec">
 <h3>📊 Backtest Doğrulama <small style="color:#8b949e;font-weight:400"
   id="valsrc"></small></h3>
 <div id="valbody"><div class="empty">doğrulanacak backtest yok —
   <code>python main.py backtest --days 30 --save-trades bt.json</code></div></div>
</section>

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

// --- backtest validation -------------------------------------------------
const pct = v => (v*100).toFixed(1) + '%';

// Confidence-interval bar: grey track = full span drawn, blue = the interval,
// white tick = point estimate, grey tick = the reference value (0 or 1).
function ciBar(lo, hi, point, ref){
  const min = Math.min(lo, ref, point), max = Math.max(hi, ref, point);
  const span = (max - min) || 1;
  const x = v => ((v - min) / span * 100).toFixed(1);
  const w = ((hi - lo) / span * 100).toFixed(1);
  return `<div class="cibar"><i style="left:${x(lo)}%;width:${w}%"></i>
          <u style="left:${x(ref)}%"></u><b style="left:${x(point)}%"></b></div>`;
}

function equitySvg(curve){
  const W = 900, H = 190, P = 4;
  const lo = Math.min(...curve), hi = Math.max(...curve);
  const span = (hi - lo) || 1;
  const x = i => P + i / (curve.length - 1) * (W - 2*P);
  const y = v => P + (1 - (v - lo) / span) * (H - 2*P);
  const line = curve.map((v,i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const zero = (lo <= 0 && hi >= 0)
    ? `<line x1="${P}" x2="${W-P}" y1="${y(0).toFixed(1)}" y2="${y(0).toFixed(1)}"
             stroke="#8b949e" stroke-dasharray="4 4" stroke-width="1"/>` : '';
  const up = curve[curve.length-1] >= curve[0];
  const col = up ? '#3fb950' : '#f85149';
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"
            style="width:100%;height:190px;background:#161b22;border:1px solid #30363d;
                   border-radius:8px">
      ${zero}
      <polyline points="${line}" fill="none" stroke="${col}" stroke-width="2"
        stroke-linejoin="round"/></svg>`;
}

async function refreshVal(){
  const v = await (await fetch('/api/validation')).json();
  const body = document.getElementById('valbody');
  const src = document.getElementById('valsrc');
  if (!v.available){
    src.textContent = '';
    body.innerHTML = `<div class="empty">${v.hint || 'veri yok'}</div>`;
    return;
  }
  src.textContent = `— ${v.source} · ${v.n_trades} işlem · ${v.n_days} seans `
                  + `(${v.span}) · ${v.setups.join(', ')}`;
  const e = v.expectancy, pf = v.profit_factor, wr = v.win_rate;
  body.innerHTML = `
    <div class="verdict ${v.passed?'pass':'fail'}">${v.passed?'✅':'⛔'} ${v.headline}
      ${v.notes.length ? '<ul>' + v.notes.map(n=>`<li>${n}</li>`).join('') + '</ul>' : ''}
    </div>
    <div class="cards">
      <div class="card"><div class="lbl">Beklenti (R/işlem)</div>
        <div class="val" style="color:${e.lo>0?'#3fb950':'#e3b341'}">
          ${e.point>=0?'+':''}${e.point.toFixed(3)}</div>
        <div class="ci">%95 GA ${e.lo.toFixed(3)} … ${e.hi.toFixed(3)}</div>
        ${ciBar(e.lo, e.hi, e.point, 0)}</div>
      <div class="card"><div class="lbl">Profit factor</div>
        <div class="val" style="color:${pf.lo>1?'#3fb950':'#e3b341'}">
          ${pf.point.toFixed(2)}</div>
        <div class="ci">%95 GA ${pf.lo.toFixed(2)} … ${pf.hi.toFixed(2)}</div>
        ${ciBar(pf.lo, pf.hi, pf.point, 1)}</div>
      <div class="card"><div class="lbl">Win rate</div>
        <div class="val">${pct(wr.point)}</div>
        <div class="ci">%95 GA ${pct(wr.lo)} … ${pct(wr.hi)}</div>
        ${ciBar(wr.lo, wr.hi, wr.point, 0.5)}</div>
      <div class="card"><div class="lbl">Toplam PnL</div>
        <div class="val ${v.total_pnl>=0?'pos':'neg'}">
          ${v.total_pnl>=0?'+':''}$${Math.abs(v.total_pnl).toFixed(2)}</div>
        <div class="ci">koşu sonu GA $${v.final_pnl_ci.lo.toFixed(0)} …
          $${v.final_pnl_ci.hi.toFixed(0)}</div></div>
      <div class="card"><div class="lbl">Max drawdown</div>
        <div class="val neg">$${v.max_dd.toFixed(2)}</div>
        <div class="ci">%95 GA $${v.max_dd_ci.lo.toFixed(0)} …
          $${v.max_dd_ci.hi.toFixed(0)}</div></div>
      <div class="card"><div class="lbl">Zararla bitme riski</div>
        <div class="val" style="color:${v.p_losing_run>0.05?'#f85149':'#3fb950'}">
          ${pct(v.p_losing_run)}</div>
        <div class="ci">DD > sermaye %${v.dd_limit_pct.toFixed(0)}:
          ${pct(v.p_deep_dd)}</div></div>
      <div class="card"><div class="lbl">Edge yok olasılığı</div>
        <div class="val" style="color:${v.p_no_edge>0.05?'#f85149':'#3fb950'}">
          ${pct(v.p_no_edge)}</div>
        <div class="ci">${v.trades_needed
            ? 'kanıt için ~' + v.trades_needed.toLocaleString() + ' işlem gerekir'
            : 'beklenti pozitif değil'}</div></div>
    </div>
    <div style="color:#8b949e;font-size:11px;margin-bottom:6px">
      Equity curve (kümülatif PnL, işlem sırasına göre)</div>
    ${equitySvg(v.equity)}`;
}

refresh(); setInterval(refresh, 5000);
refreshVal(); setInterval(refreshVal, 15000);
</script></body></html>"""
