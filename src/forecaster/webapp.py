"""Web UI: pick symbols, trigger analysis on demand, see results (FastAPI, single page)."""
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
    timeframe: str | None = None
    profile: str | None = None
    news_sources: list[str] | None = None


class AnalyzeRequest(BaseModel):
    symbols: list[SymbolIn]


class WatchlistItemIn(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    notes: str | None = None
    sources: str = "google"
    timeframes: str = "1d"
    profiles: str = "balanced"


def _prediction_to_dict(p: Prediction) -> dict:
    return {
        "ts": p.ts.isoformat(),
        "symbol": p.symbol,
        "timeframe": p.timeframe,
        "profile": p.profile,
        "news_sources": p.news_sources,
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
        except Exception as exc:
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
            return
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

    @app.get("/api/watchlist")
    def api_watchlist(_: None = Depends(require_auth)) -> JSONResponse:
        recorder = PredictionRecorder(cfg.db_path)
        try:
            return JSONResponse([dict(row) for row in recorder.list_watchlist()])
        finally:
            recorder.close()

    @app.post("/api/watchlist")
    def api_watchlist_upsert(item: WatchlistItemIn, _: None = Depends(require_auth)) -> JSONResponse:
        recorder = PredictionRecorder(cfg.db_path)
        try:
            recorder.upsert_watchlist(
                item.symbol, item.name, item.sector, item.notes,
                item.sources, item.timeframes, item.profiles,
            )
            return JSONResponse({"ok": True})
        finally:
            recorder.close()

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest, _: None = Depends(require_auth)) -> JSONResponse:
        if len(req.symbols) > cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {cfg.max_symbols_per_run} sembol seçebilirsiniz.",
            )
        symbols = [{
            "symbol": s.symbol,
            "name": s.name,
            "timeframe": s.timeframe or "1d",
            "profile": s.profile or "balanced",
            "news_sources": s.news_sources or ["google"],
        } for s in req.symbols]
        started = state.start_analysis(symbols)
        return JSONResponse({"started": started})

    @app.post("/api/analyze/multi")
    def api_analyze_multi(req: AnalyzeRequest, _: None = Depends(require_auth)) -> JSONResponse:
        if len(req.symbols) > cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {cfg.max_symbols_per_run} sembol seçebilirsiniz.",
            )
        symbols = [
            {
                "symbol": s.symbol,
                "name": s.name,
                "timeframe": timeframe,
                "profile": s.profile or "balanced",
                "news_sources": s.news_sources or ["google"],
            }
            for s in req.symbols
            for timeframe in (s.timeframe.split(",") if s.timeframe else ["1d"])
        ]
        started = state.start_analysis(symbols)
        return JSONResponse({"started": started, "runs": len(symbols)})

    @app.post("/api/analyze/compare")
    def api_analyze_compare(req: AnalyzeRequest, _: None = Depends(require_auth)) -> JSONResponse:
        if len(req.symbols) > cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {cfg.max_symbols_per_run} sembol seçebilirsiniz.",
            )
        profiles = ["balanced", "news_heavy", "technical_heavy"]
        symbols = [
            {
                "symbol": s.symbol,
                "name": s.name,
                "timeframe": s.timeframe or "1d",
                "profile": profile,
                "news_sources": s.news_sources or ["google"],
            }
            for s in req.symbols
            for profile in profiles
        ]
        started = state.start_analysis(symbols)
        return JSONResponse({"started": started, "comparisons": len(symbols)})

    @app.get("/api/state")
    def api_state(_: None = Depends(require_auth)) -> JSONResponse:
        with state.lock:
            return JSONResponse({
                "status": state.status,
                "progress": state.progress,
                "error": state.error,
                "results": state.results,
            })

    @app.get("/api/dashboard")
    def api_dashboard(days: int = 30, _: None = Depends(require_auth)) -> JSONResponse:
        recorder = PredictionRecorder(cfg.db_path)
        try:
            watchlist = [dict(row) for row in recorder.list_watchlist()]
            recent = [dict(row) for row in recorder.recent(limit=50)]
            hit_hits, hit_total = recorder.hit_rate(days)
            by_profile = [dict(row) for row in recorder.summary_by_profile(days)]
            by_timeframe = [dict(row) for row in recorder.summary_by_timeframe(days)]
            by_direction = [dict(row) for row in recorder.summary_by_direction(days)]
            by_symbol = [dict(row) for row in recorder.summary_by_symbol(days, limit=10)]
        finally:
            recorder.close()

        hit_series = []
        running_hits = 0
        running_total = 0
        for row in reversed(recent):
            running_total += 1
            running_hits += 1 if row.get("hit") else 0
            hit_series.append({
                "ts": row["ts"],
                "symbol": row["symbol"],
                "running_hit_rate": round((running_hits / running_total) * 100, 1),
            })

        return JSONResponse({
            "watchlist": watchlist,
            "hit_rate": {"hits": hit_hits, "total": hit_total},
            "by_profile": by_profile,
            "by_timeframe": by_timeframe,
            "by_direction": by_direction,
            "by_symbol": by_symbol,
            "hit_series": hit_series[-30:],
            "recent": recent,
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
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TradeView Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#07111f;--bg2:#0c1629;--panel:#101b30;--panel2:#13213a;--text:#e7eefb;--muted:#8ea4c7;--line:#223557;--green:#31c48d;--red:#ff6b6b;--amber:#f4b942;--blue:#6ca7ff}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:14px/1.55 Inter,Segoe UI,sans-serif;color:var(--text);background:radial-gradient(circle at top left, rgba(108,167,255,.18), transparent 28%),radial-gradient(circle at top right, rgba(49,196,141,.12), transparent 22%),linear-gradient(180deg,var(--bg),var(--bg2))}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 40px}
.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;flex-wrap:wrap;margin-bottom:18px}
h1{margin:0;font-size:30px;letter-spacing:-.02em}
.sub{color:var(--muted);margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.03)}
.pill{display:inline-flex;align-items:center;padding:2px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--line);color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.015));border:1px solid var(--line);border-radius:18px;padding:16px;box-shadow:0 18px 40px rgba(0,0,0,.22)}
.stat{grid-column:span 3}.stat .v{font-size:26px;font-weight:700;margin-top:8px}.stat .l{color:var(--muted);font-size:12px}
.panel{grid-column:span 12}.half{grid-column:span 6}
.search{position:relative;grid-column:span 12}
input[type=text]{width:100%;background:rgba(255,255,255,.04);color:var(--text);border:1px solid var(--line);border-radius:14px;padding:14px 14px;font-size:14px;outline:none}
input[type=text]:focus{border-color:#35558b;box-shadow:0 0 0 4px rgba(108,167,255,.12)}
.dropdown{position:absolute;left:0;right:0;top:100%;margin-top:8px;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;z-index:20;max-height:260px;overflow-y:auto}
.dropdown div{padding:10px 14px;cursor:pointer;border-bottom:1px solid rgba(255,255,255,.04)}
.dropdown div:hover{background:rgba(255,255,255,.04)}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.chips{display:flex;flex-wrap:wrap;gap:8px;min-height:28px}
.chip{display:inline-flex;align-items:center;gap:8px;padding:7px 11px;border-radius:999px;background:rgba(255,255,255,.05);border:1px solid var(--line)}
.chip button,.btn{border:0;cursor:pointer;border-radius:12px}.chip button{background:transparent;color:var(--muted);padding:0;font-size:14px}
.btn{padding:10px 14px;font-weight:600;color:#06111e;background:var(--blue)}.btn.alt{background:rgba(255,255,255,.08);color:var(--text);border:1px solid var(--line)}.btn.good{background:var(--green)}.btn.warn{background:var(--amber)}.btn.danger{background:var(--red)}
.btn:disabled{opacity:.55;cursor:not-allowed}
table{width:100%;border-collapse:collapse}th,td{padding:11px 10px;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top;text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}td{color:#dce7f8}
.up{color:var(--green);font-weight:700}.down{color:var(--red);font-weight:700}.neutral{color:var(--muted);font-weight:700}.reasons,.muted{color:var(--muted)}.empty{color:var(--muted);padding:18px 10px;font-style:italic;text-align:center}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:999px;display:inline-block}
canvas{width:100%!important;height:320px!important}
@media (max-width:1000px){.stat,.half{grid-column:span 12}}
</style></head><body>
<div class="wrap">
  <div class="hero">
    <div>
      <h1>TradeView</h1>
      <div class="sub">Haber + teknik analiz, Groq destekli yorumlama, watchlist yönetimi ve performans dashboard'u.</div>
    </div>
    <div class="badge"><span class="pill">Groq: llama-3.3-70b-versatile</span><span id="status" class="pill">idle</span></div>
  </div>

  <div class="grid">
    <div class="card stat"><div class="l">Son 30 gün hit rate</div><div id="statHit" class="v">-</div></div>
    <div class="card stat"><div class="l">Watchlist</div><div id="statWatchlist" class="v">-</div></div>
    <div class="card stat"><div class="l">Son model grubu</div><div id="statModel" class="v">-</div></div>
    <div class="card stat"><div class="l">Son zaman dilimi</div><div id="statTf" class="v">-</div></div>

    <div class="card search">
      <div class="row" style="justify-content:space-between;margin-bottom:10px">
        <div class="muted">Sembol veya şirket ara, seç ve analiz et.</div>
        <div class="legend"><span><i class="dot" style="background:var(--green)"></i> UP</span><span><i class="dot" style="background:var(--red)"></i> DOWN</span><span><i class="dot" style="background:var(--muted)"></i> NEUTRAL</span></div>
      </div>
      <input type="text" id="q" placeholder="AAPL, Apple, MSFT..." autocomplete="off">
      <div id="dd" class="dropdown" style="display:none"></div>
      <div class="row" style="margin-top:12px;justify-content:space-between">
        <div class="chips" id="chips"></div>
        <div class="row">
          <button class="btn good" id="go" onclick="analyze()">Analiz Et</button>
          <button class="btn alt" onclick="compareModels()">Model Karşılaştır</button>
          <button class="btn alt" onclick="multiTimeframe()">Çok Zaman Dilimi</button>
          <button class="btn alt" onclick="saveWatchlist()">Watchlist'e Kaydet</button>
          <button class="btn alt" onclick="loadDashboard()">Dashboard Yenile</button>
        </div>
      </div>
      <div id="progress" class="sub" style="margin-top:10px"></div>
    </div>

    <div class="card half"><h3 style="margin:0 0 10px">Performans</h3><canvas id="hitChart"></canvas></div>
    <div class="card half"><h3 style="margin:0 0 10px">Model / Zaman Dilimi</h3><canvas id="barChart"></canvas></div>

    <div class="card panel"><h3 style="margin:0 0 10px">Son Analiz Sonuçları</h3>
      <table><thead><tr><th>Sembol</th><th>Zaman</th><th>Profil</th><th>Yön</th><th>Final</th><th>Güven</th><th>Haber</th><th>Teknik</th><th>Detay</th></tr></thead><tbody id="results"><tr><td colspan="9" class="empty">Henüz analiz yok</td></tr></tbody></table>
    </div>

    <div class="card half"><h3 style="margin:0 0 10px">Watchlist</h3><table><thead><tr><th>Sembol</th><th>Ad</th><th>Sinyal</th><th>Detay</th></tr></thead><tbody id="watchlist"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table></div>
    <div class="card half"><h3 style="margin:0 0 10px">Güncel İzleme</h3><table><thead><tr><th>Zaman</th><th>Sembol</th><th>Yön</th><th>İsabet</th></tr></thead><tbody id="history"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table></div>
  </div>
</div>

<script>
let selected = [];
let pollTimer = null;
let hitChart = null;
let barChart = null;

function dirClass(d){ return d==='UP' ? 'up' : (d==='DOWN' ? 'down' : 'neutral'); }
function pct(hit, total){ return total ? Math.round(100 * hit / total) + '%' : '0%'; }

const qEl = document.getElementById('q');
qEl.addEventListener('input', async () => {
  const q = qEl.value.trim();
  const dd = document.getElementById('dd');
  if (q.length < 2) { dd.style.display = 'none'; return; }
  const r = await fetch('/api/symbols?q=' + encodeURIComponent(q));
  const items = await r.json();
  if (!items.length) { dd.style.display = 'none'; return; }
  dd.innerHTML = items.map(it => `<div onclick='pick(${JSON.stringify(it).replace(/'/g,"&#39;")})'><b>${it.symbol}</b> — ${it.name}</div>`).join('');
  dd.style.display = 'block';
});

function pick(item){
  if (!selected.some(s => s.symbol === item.symbol)) selected.push({...item, timeframe:'1d,1wk,1mo', profile:'balanced', news_sources:['google']});
  qEl.value = '';
  document.getElementById('dd').style.display = 'none';
  renderChips();
}

function remove(symbol){ selected = selected.filter(s => s.symbol !== symbol); renderChips(); }
function renderChips(){ document.getElementById('chips').innerHTML = selected.map(s => `<span class="chip">${s.symbol}<button onclick="remove('${s.symbol}')">×</button></span>`).join(''); }

async function postAnalyze(url, payload){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (!r.ok){
    const err = await r.json().catch(() => ({}));
    alert(err.detail || 'İşlem başlatılamadı.');
    return false;
  }
  const data = await r.json();
  if (data.started) startPolling();
  return true;
}

function selectedPayload(){
  return {symbols: selected.map(s => ({symbol:s.symbol, name:s.name, timeframe:s.timeframe, profile:s.profile, news_sources:s.news_sources}))};
}

async function analyze(){ if (selected.length) await postAnalyze('/api/analyze', selectedPayload()); }
async function compareModels(){ if (selected.length) await postAnalyze('/api/analyze/compare', selectedPayload()); }
async function multiTimeframe(){ if (selected.length) await postAnalyze('/api/analyze/multi', selectedPayload()); }

async function saveWatchlist(){
  if (!selected.length) return;
  for (const s of selected){
    await fetch('/api/watchlist', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({symbol:s.symbol, name:s.name, sources:(s.news_sources || ['google']).join(','), timeframes:s.timeframe, profiles:s.profile}),
    });
  }
  loadDashboard();
}

function startPolling(){ if (pollTimer) return; pollTimer = setInterval(refreshState, 1500); refreshState(); }

async function refreshState(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const st = document.getElementById('status');
  st.textContent = s.status + (s.progress ? ' · ' + s.progress : '');
  st.className = 'pill ' + (s.status === 'error' ? 'danger' : (s.status === 'running' ? 'warn' : 'good'));
  document.getElementById('go').disabled = (s.status === 'running');
  document.getElementById('progress').textContent = s.error ? ('Hata: ' + s.error) : (s.progress || '');

  const results = document.getElementById('results');
  if (s.results.length){
    results.innerHTML = s.results.map(p => `<tr>
      <td><b>${p.symbol}</b></td>
      <td>${p.timeframe}</td>
      <td>${p.profile}</td>
      <td class="${dirClass(p.final_direction)}">${p.final_direction}</td>
      <td>${p.final_score.toFixed(2)}</td>
      <td>${p.final_confidence.toFixed(2)}</td>
      <td>${p.news_score.toFixed(2)} (${p.news_confidence.toFixed(2)})</td>
      <td>${p.technical_score.toFixed(2)}</td>
      <td class="reasons">${p.news_rationale}<br>${p.technical_reasons.join('; ')}</td>
      </tr>`).join('');
  }
  if (s.status !== 'running' && pollTimer){ clearInterval(pollTimer); pollTimer = null; loadHistory(); loadDashboard(); }
}

async function loadHistory(){
  const r = await fetch('/api/history?days=7');
  const s = await r.json();
  const hr = s.hit_rate;
  document.getElementById('statHit').textContent = pct(hr.hits, hr.total);
  document.getElementById('history').innerHTML = s.recent.length ? s.recent.map(p => `<tr>
    <td>${new Date(p.ts).toLocaleString()}</td>
    <td><b>${p.symbol}</b></td>
    <td class="${dirClass(p.final_direction)}">${p.final_direction}</td>
    <td>${p.hit === null ? '—' : (p.hit ? '✅' : '❌')}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty">Henüz tahmin yok</td></tr>';
}

function chartOrUpdate(current, ctx, config){ if (current) current.destroy(); return new Chart(ctx, config); }

async function loadDashboard(){
  const r = await fetch('/api/dashboard?days=30');
  const d = await r.json();
  document.getElementById('statHit').textContent = d.hit_rate.total > 0 ? `${d.hit_rate.hits}/${d.hit_rate.total} (${pct(d.hit_rate.hits, d.hit_rate.total)})` : 'Veri yok';
  document.getElementById('statWatchlist').textContent = d.watchlist.length;
  document.getElementById('statModel').textContent = d.by_profile[0] ? d.by_profile[0].profile : '-';
  document.getElementById('statTf').textContent = d.by_timeframe[0] ? d.by_timeframe[0].timeframe : '-';

  hitChart = chartOrUpdate(hitChart, document.getElementById('hitChart'), {
    type:'line',
    data:{ labels:d.hit_series.map(x => new Date(x.ts).toLocaleDateString()), datasets:[{label:'Running hit rate %', data:d.hit_series.map(x => x.running_hit_rate), borderColor:'#6ca7ff', backgroundColor:'rgba(108,167,255,.18)', tension:.3, fill:true }]},
    options:{ responsive:true, plugins:{ legend:{display:false} }, scales:{ y:{ beginAtZero:true, max:100, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
  });

  const profileMap = new Map((d.by_profile || []).map(x => [x.profile, x.hits]));
  const timeframeMap = new Map((d.by_timeframe || []).map(x => [x.timeframe, x.hits]));
  const labels = [...new Set([...profileMap.keys(), ...timeframeMap.keys()])].filter(Boolean);
  barChart = chartOrUpdate(barChart, document.getElementById('barChart'), {
    type:'bar',
    data:{ labels, datasets:[{label:'Profile hits', data:labels.map(l => profileMap.get(l) || 0), backgroundColor:'#31c48d'}, {label:'Timeframe hits', data:labels.map(l => timeframeMap.get(l) || 0), backgroundColor:'#6ca7ff'}] },
    options:{ responsive:true, plugins:{ legend:{ labels:{ color:'#e7eefb' } } }, scales:{ y:{ beginAtZero:true, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
  });

  document.getElementById('watchlist').innerHTML = d.watchlist.length ? d.watchlist.map(w => `<tr><td><b>${w.symbol}</b></td><td>${w.name || ''}</td><td>${w.profiles || ''} / ${w.timeframes || ''}</td><td class="muted">${w.sources || ''}</td></tr>`).join('') : '<tr><td colspan="4" class="empty">Watchlist boş</td></tr>';
}

async function loadWatchlist(){
  const r = await fetch('/api/watchlist');
  const items = await r.json();
  if (!items.length) return;
  document.getElementById('watchlist').innerHTML = items.map(w => `<tr><td><b>${w.symbol}</b></td><td>${w.name || ''}</td><td>${w.profiles || ''} / ${w.timeframes || ''}</td><td class="muted">${w.sources || ''}</td></tr>`).join('');
}

loadWatchlist();
loadHistory();
loadDashboard();
</script></body></html>"""