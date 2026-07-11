"""Web UI: pick symbols, trigger analysis on demand, see results (FastAPI, single page).

Run:  python main.py serve  ->  http://localhost:8000
"""
from __future__ import annotations

import csv
import io
import logging
import secrets
import threading

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

from .config import Config
from .models import Prediction
from .pipeline import load_watchlist, run_for_symbols
from .storage.recorder import PredictionRecorder
from .symbols_search import search_symbols

log = logging.getLogger(__name__)


class SymbolIn(BaseModel):
    symbol: str
    name: str | None = None


class AnalyzeRequest(BaseModel):
    symbols: list[SymbolIn]


def _prediction_to_dict(p: Prediction) -> dict:
    return {
        "ts": p.ts.isoformat(),
        "symbol": p.symbol,
        "news_score": round(p.news_score, 3),
        "news_confidence": round(p.news_confidence, 3),
        "news_rationale": p.news_rationale,
        "technical_score": round(p.technical_score, 3),
        "technical_reasons": p.technical_reasons,
        "final_score": round(p.final_score, 3),
        "final_direction": p.final_direction.value,
        "final_confidence": round(p.final_confidence, 3),
        "price_at_prediction": p.price_at_prediction,
    }


class AnalysisState:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.lock = threading.Lock()
        self.status = "idle"
        self.progress = ""
        self.results: list[dict] = []
        self.error: str | None = None

    def start_analysis(self, symbols: list[dict]) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.progress = ""
            self.error = None
        threading.Thread(target=self._run, args=(symbols,), daemon=True).start()
        return True

    def _run(self, symbols: list[dict]) -> None:
        try:
            predictions = run_for_symbols(symbols, self.cfg, progress_cb=self._set_progress)
            with self.lock:
                self.results = [_prediction_to_dict(p) for p in predictions]
                self.status = "done"
                self.progress = ""
        except Exception as exc:                # surface any failure in the UI
            log.exception("analysis failed")
            with self.lock:
                self.status = "error"
                self.error = str(exc)

    def _set_progress(self, msg: str) -> None:
        with self.lock:
            self.progress = msg


def create_app(cfg: Config) -> FastAPI:
    app = FastAPI(title="Haber + Teknik Analiz Tahmin Sistemi")
    state = AnalysisState(cfg)
    security = HTTPBasic()

    def require_auth(credentials: HTTPBasicCredentials = Depends(security)) -> None:
        if not cfg.app_password:
            return  # no password configured: auth disabled (local dev)
        if not secrets.compare_digest(credentials.password, cfg.app_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Basic"},
            )

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/symbols")
    def api_symbols(q: str = "", _: None = Depends(require_auth)) -> JSONResponse:
        return JSONResponse(search_symbols(q))

    @app.get("/api/favorites")
    def api_favorites(_: None = Depends(require_auth)) -> JSONResponse:
        try:
            return JSONResponse(load_watchlist(cfg.watchlist_path))
        except (OSError, ValueError):
            return JSONResponse([])

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest, _: None = Depends(require_auth)) -> JSONResponse:
        if len(req.symbols) > cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {cfg.max_symbols_per_run} sembol seçebilirsiniz.",
            )
        symbols = [{"symbol": s.symbol, "name": s.name} for s in req.symbols]
        started = state.start_analysis(symbols)
        return JSONResponse({"started": started})

    @app.get("/api/state")
    def api_state(_: None = Depends(require_auth)) -> JSONResponse:
        with state.lock:
            return JSONResponse({
                "status": state.status,
                "progress": state.progress,
                "error": state.error,
                "results": state.results,
            })

    @app.get("/api/history")
    def api_history(days: int = 7, _: None = Depends(require_auth)) -> JSONResponse:
        recorder = PredictionRecorder(cfg.db_path)
        try:
            rows = recorder.recent(limit=50)
            hits, total = recorder.hit_rate(days)
        finally:
            recorder.close()
        return JSONResponse({
            "hit_rate": {"hits": hits, "total": total},
            "recent": [dict(r) for r in rows],
        })

    @app.get("/api/history.csv")
    def api_history_csv(_: None = Depends(require_auth)) -> PlainTextResponse:
        recorder = PredictionRecorder(cfg.db_path)
        try:
            rows = recorder.recent(limit=500)
        finally:
            recorder.close()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ts", "symbol", "final_score", "final_direction",
                          "final_confidence", "actual_direction", "hit"])
        for r in rows:
            writer.writerow([r["ts"], r["symbol"], r["final_score"], r["final_direction"],
                              r["final_confidence"], r["actual_direction"], r["hit"]])
        return PlainTextResponse(
            buf.getvalue(), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )

    @app.get("/", response_class=HTMLResponse)
    def index(_: None = Depends(require_auth)) -> str:
        return PAGE

    return app


PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8">
<title>Haber + Teknik Analiz Tahmin Sistemi</title>
<style>
 :root { color-scheme: dark; }
 body { background:#0d1117; color:#e6edf3; font:14px/1.5 'Segoe UI',sans-serif;
        margin:0; padding:24px; max-width:1000px; }
 h1 { font-size:20px; margin:0 0 4px; }
 .sub { color:#8b949e; font-size:12px; margin-bottom:16px; }
 .search-box { position:relative; margin-bottom:12px; }
 input[type=text] { width:100%; box-sizing:border-box; background:#161b22; color:#e6edf3;
        border:1px solid #30363d; border-radius:6px; padding:10px 12px; font-size:14px; }
 .dropdown { position:absolute; top:100%; left:0; right:0; background:#161b22;
        border:1px solid #30363d; border-radius:6px; margin-top:4px; z-index:10;
        max-height:240px; overflow-y:auto; }
 .dropdown div { padding:8px 12px; cursor:pointer; }
 .dropdown div:hover { background:#21262d; }
 .chips { display:flex; flex-wrap:wrap; gap:8px; margin:12px 0; }
 .chip { background:#21262d; border:1px solid #30363d; border-radius:14px;
        padding:4px 10px 4px 12px; font-size:13px; display:flex; align-items:center; gap:6px; }
 .chip button { background:none; border:0; color:#8b949e; cursor:pointer; font-size:14px; padding:0; }
 button.primary { background:#238636; color:#fff; border:0; border-radius:6px;
          padding:9px 20px; font-weight:600; cursor:pointer; }
 button.primary:disabled { background:#30363d; cursor:wait; }
 .badge { display:inline-block; padding:2px 10px; border-radius:12px;
          font-size:11px; font-weight:600; margin-left:8px; }
 .warn { background:#3d2e00; color:#e3b341; border:1px solid #e3b341; }
 .ok   { background:#0f2e1b; color:#3fb950; border:1px solid #3fb950; }
 .err  { background:#3d0d0d; color:#f85149; border:1px solid #f85149; }
 table { border-collapse:collapse; width:100%; margin-top:16px; }
 th { text-align:left; color:#8b949e; font-size:11px; text-transform:uppercase;
      border-bottom:1px solid #30363d; padding:6px 10px; }
 td { padding:8px 10px; border-bottom:1px solid #21262d; vertical-align:top; }
 tr:hover td { background:#161b22; }
 .up { color:#3fb950; font-weight:700; } .down { color:#f85149; font-weight:700; }
 .neutral { color:#8b949e; font-weight:700; }
 .reasons { color:#8b949e; font-size:12px; }
 .empty { color:#8b949e; padding:18px 10px; font-style:italic; }
 .fav { background:none; border:1px solid #30363d; color:#e6edf3; border-radius:14px;
        padding:4px 12px; font-size:12px; cursor:pointer; }
 .fav:hover { background:#21262d; }
 a.csv { color:#58a6ff; font-size:12px; text-decoration:none; }
 a.csv:hover { text-decoration:underline; }
</style></head><body>
<h1>Haber + Teknik Analiz Tahmin Sistemi <span id="status" class="badge ok">idle</span></h1>
<div class="sub">Analiz etmek istediginiz hisseleri arayin ve secin, sonra "Analiz Et" butonuna basin.</div>
<div class="search-box">
 <input type="text" id="q" placeholder="Sembol veya sirket adi ara (orn. AAPL, Apple)" autocomplete="off">
 <div id="dd" class="dropdown" style="display:none"></div>
</div>
<div class="chips" id="favorites"></div>
<div class="chips" id="chips"></div>
<button class="primary" id="go" onclick="analyze()">Analiz Et</button>
<span id="progress" class="sub"></span>

<table><thead><tr><th>Sembol</th><th>Yon</th><th>Final</th><th>Guven</th>
<th>Haber</th><th>Teknik</th><th>Detay</th></tr></thead>
<tbody id="results"><tr><td colspan="7" class="empty">henuz analiz yok</td></tr></tbody></table>

<h3 style="margin-top:32px">Gecmis (son 50 tahmin) <a class="csv" href="/api/history.csv">CSV indir</a></h3>
<div class="sub" id="hitrate"></div>
<table><thead><tr><th>Zaman</th><th>Sembol</th><th>Tahmin</th><th>Gercek</th><th>Isabet</th></tr></thead>
<tbody id="history"><tr><td colspan="5" class="empty">yukleniyor...</td></tr></tbody></table>

<script>
let selected = [];
let pollTimer = null;

function dirClass(d){ return d==='UP' ? 'up' : (d==='DOWN' ? 'down' : 'neutral'); }

const qEl = document.getElementById('q');
qEl.addEventListener('input', async () => {
  const q = qEl.value.trim();
  const dd = document.getElementById('dd');
  if (q.length < 2) { dd.style.display = 'none'; return; }
  const r = await fetch('/api/symbols?q=' + encodeURIComponent(q));
  const items = await r.json();
  if (!items.length) { dd.style.display = 'none'; return; }
  dd.innerHTML = items.map(it => `<div onclick='pick(${JSON.stringify(it).replace(/'/g,"&#39;")})'>
    <b>${it.symbol}</b> — ${it.name}</div>`).join('');
  dd.style.display = 'block';
});

function pick(item){
  if (!selected.some(s => s.symbol === item.symbol)) selected.push(item);
  qEl.value = '';
  document.getElementById('dd').style.display = 'none';
  renderChips();
}
function remove(symbol){
  selected = selected.filter(s => s.symbol !== symbol);
  renderChips();
}
function renderChips(){
  document.getElementById('chips').innerHTML = selected.map(s =>
    `<span class="chip">${s.symbol}<button onclick="remove('${s.symbol}')">×</button></span>`).join('');
}

async function analyze(){
  if (!selected.length) return;
  const r = await fetch('/api/analyze', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({symbols: selected}),
  });
  if (!r.ok){
    const err = await r.json().catch(() => ({}));
    alert(err.detail || 'Analiz baslatilamadi.');
    return;
  }
  const data = await r.json();
  if (data.started) startPolling();
}

function startPolling(){
  if (pollTimer) return;
  pollTimer = setInterval(refreshState, 1500);
  refreshState();
}

async function refreshState(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const st = document.getElementById('status');
  st.textContent = s.status + (s.progress ? ' · ' + s.progress : '');
  st.className = 'badge ' + (s.status === 'error' ? 'err' : (s.status === 'running' ? 'warn' : 'ok'));
  document.getElementById('go').disabled = (s.status === 'running');
  if (s.error) st.textContent += ' — ' + s.error;

  const results = document.getElementById('results');
  if (s.results.length){
    results.innerHTML = s.results.map(p => `<tr>
      <td><b>${p.symbol}</b></td>
      <td class="${dirClass(p.final_direction)}">${p.final_direction}</td>
      <td>${p.final_score.toFixed(2)}</td>
      <td>${p.final_confidence.toFixed(2)}</td>
      <td>${p.news_score.toFixed(2)} (${p.news_confidence.toFixed(2)})</td>
      <td>${p.technical_score.toFixed(2)}</td>
      <td class="reasons">${p.news_rationale}<br>${p.technical_reasons.join('; ')}</td>
      </tr>`).join('');
  }
  if (s.status !== 'running' && pollTimer){
    clearInterval(pollTimer); pollTimer = null;
    loadHistory();
  }
}

async function loadHistory(){
  const r = await fetch('/api/history?days=7');
  const s = await r.json();
  const hr = s.hit_rate;
  document.getElementById('hitrate').textContent = hr.total > 0
    ? `Son 7 gun isabet orani: ${hr.hits}/${hr.total} (${Math.round(100*hr.hits/hr.total)}%)`
    : 'Son 7 gunde henuz isabet-onaylanmis tahmin yok.';
  const hist = document.getElementById('history');
  if (s.recent.length){
    hist.innerHTML = s.recent.map(p => `<tr>
      <td>${new Date(p.ts).toLocaleString()}</td>
      <td><b>${p.symbol}</b></td>
      <td class="${dirClass(p.final_direction)}">${p.final_direction}</td>
      <td>${p.actual_direction ?? '—'}</td>
      <td>${p.hit === null ? '—' : (p.hit ? '✅' : '❌')}</td>
      </tr>`).join('');
  } else {
    hist.innerHTML = '<tr><td colspan="5" class="empty">henuz tahmin yok</td></tr>';
  }
}

async function loadFavorites(){
  const r = await fetch('/api/favorites');
  const items = await r.json();
  document.getElementById('favorites').innerHTML = items.map(it =>
    `<button class="fav" onclick='pick(${JSON.stringify(it).replace(/'/g,"&#39;")})'>+ ${it.symbol}</button>`
  ).join('');
}

loadFavorites();
loadHistory();
</script></body></html>"""
