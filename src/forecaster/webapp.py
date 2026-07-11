"""Web UI: pick symbols, trigger analysis on demand, see results (FastAPI, single page)."""
from __future__ import annotations

import csv
import io
import logging
import secrets
import threading
from dataclasses import replace

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .config import Config
from .models import Prediction
from .pipeline import load_watchlist, run_for_symbols
from .storage.recorder import PredictionRecorder
from .symbols_search import search_symbols

log = logging.getLogger(__name__)
COOKIE_NAME = "tradeview_session"


class SymbolIn(BaseModel):
    symbol: str
    name: str | None = None
    timeframe: str | None = None
    profile: str | None = None
    news_sources: list[str] | None = None


class AnalyzeRequest(BaseModel):
    symbols: list[SymbolIn]


class AuthRequest(BaseModel):
    username: str
    password: str


class WatchlistItemIn(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    notes: str | None = None
    sources: str = "google"
    timeframes: str = "1d"
    profiles: str = "balanced"


class AppSettingsIn(BaseModel):
    news_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    technical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_band: float | None = Field(default=None, ge=0.0, le=1.0)
    groq_model: str | None = None
    news_lookback_hours: int | None = Field(default=None, ge=1, le=168)
    max_articles_per_symbol: int | None = Field(default=None, ge=1, le=50)
    max_symbols_per_run: int | None = Field(default=None, ge=1, le=100)
    intraday_lookback_period: str | None = None
    technical_lookback_period: str | None = None


_SETTING_TYPES: dict[str, type] = {
    "news_weight": float,
    "technical_weight": float,
    "neutral_band": float,
    "groq_model": str,
    "news_lookback_hours": int,
    "max_articles_per_symbol": int,
    "max_symbols_per_run": int,
    "intraday_lookback_period": str,
    "technical_lookback_period": str,
}


def _coerce_settings(raw: dict[str, str]) -> dict[str, object]:
    coerced: dict[str, object] = {}
    for key, expected_type in _SETTING_TYPES.items():
        value = raw.get(key)
        if value is None:
            continue
        coerced[key] = expected_type(value)
    return coerced


class RuntimeState:
    def __init__(self, base_cfg: Config, initial_settings: dict[str, str]) -> None:
        self.base_cfg = base_cfg
        self.lock = threading.Lock()
        self.settings = dict(initial_settings)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return _coerce_settings(self.settings)

    def update(self, updates: dict[str, object]) -> dict[str, object]:
        with self.lock:
            for key, value in updates.items():
                self.settings[key] = str(value)
            return _coerce_settings(self.settings)

    def current_cfg(self) -> Config:
        with self.lock:
            overrides = _coerce_settings(self.settings)
        return replace(self.base_cfg, **overrides)


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
    def __init__(self, runtime: RuntimeState) -> None:
        self.runtime = runtime
        self.lock = threading.Lock()
        self.status = "idle"
        self.progress = ""
        self.results: list[dict] = []
        self.error: str | None = None

    def start_analysis(self, symbols: list[dict], user_id: int | None = None) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.progress = ""
            self.error = None
        threading.Thread(target=self._run, args=(symbols, user_id), daemon=True).start()
        return True

    def _run(self, symbols: list[dict], user_id: int | None) -> None:
        try:
            predictions = run_for_symbols(
                symbols,
                self.runtime.current_cfg(),
                progress_cb=self._set_progress,
                user_id=user_id,
            )
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
    runtime_cache: dict[int, RuntimeState] = {}
    state_cache: dict[int, AnalysisState] = {}

    def _recorder() -> PredictionRecorder:
        return PredictionRecorder(cfg.db_path)

    def _get_user_id_from_request(request: Request) -> int | None:
        token = request.cookies.get(COOKIE_NAME)
        if not token:
            return None
        recorder = _recorder()
        try:
            return recorder.get_session_user_id(token)
        finally:
            recorder.close()

    def require_user(request: Request) -> int:
        user_id = _get_user_id_from_request(request)
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
        return user_id

    def _get_runtime(user_id: int) -> RuntimeState:
        runtime = runtime_cache.get(user_id)
        if runtime is None:
            recorder = _recorder()
            try:
                initial_settings = recorder.get_settings(user_id)
            finally:
                recorder.close()
            runtime = RuntimeState(cfg, initial_settings)
            runtime_cache[user_id] = runtime
        return runtime

    def _get_state(user_id: int) -> AnalysisState:
        state = state_cache.get(user_id)
        if state is None:
            state = AnalysisState(_get_runtime(user_id))
            state_cache[user_id] = state
        return state

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/api/register")
    def api_register(item: AuthRequest, response: Response) -> JSONResponse:
        username = item.username.strip().lower()
        if not username or not item.password:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")
        recorder = _recorder()
        try:
            if recorder.get_user(username) is not None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
            user_id = recorder.create_user(username, item.password)
            token = recorder.create_session(user_id)
        finally:
            recorder.close()
        response = JSONResponse({"ok": True, "username": username})
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
        return response

    @app.post("/api/login")
    def api_login(item: AuthRequest) -> JSONResponse:
        username = item.username.strip().lower()
        recorder = _recorder()
        try:
            user_id = recorder.authenticate_user(username, item.password)
            if user_id is None:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
            token = recorder.create_session(user_id)
        finally:
            recorder.close()
        response = JSONResponse({"ok": True, "username": username})
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
        return response

    @app.post("/api/logout")
    def api_logout(request: Request) -> JSONResponse:
        token = request.cookies.get(COOKIE_NAME)
        if token:
            recorder = _recorder()
            try:
                recorder.delete_session(token)
            finally:
                recorder.close()
        response = JSONResponse({"ok": True})
        response.delete_cookie(COOKIE_NAME)
        return response

    @app.get("/api/me")
    def api_me(request: Request) -> JSONResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            user = recorder.get_user_by_id(user_id)
        finally:
            recorder.close()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login required")
        return JSONResponse({"id": int(user["id"]), "username": str(user["username"])})

    @app.get("/api/symbols")
    def api_symbols(q: str = "", _: int = Depends(require_user)) -> JSONResponse:
        return JSONResponse(search_symbols(q))

    @app.get("/api/settings")
    def api_settings(request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        return JSONResponse(_get_runtime(user_id).snapshot())

    @app.put("/api/settings")
    def api_settings_update(item: AppSettingsIn, request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        updates = item.model_dump(exclude_none=True)
        if not updates:
            return JSONResponse(_get_runtime(user_id).snapshot())
        recorder = _recorder()
        try:
            recorder.upsert_settings(updates, user_id=user_id)
        finally:
            recorder.close()
        return JSONResponse(_get_runtime(user_id).update(updates))

    @app.get("/api/favorites")
    def api_favorites(_: int = Depends(require_user)) -> JSONResponse:
        try:
            return JSONResponse(load_watchlist(cfg.watchlist_path))
        except (OSError, ValueError):
            return JSONResponse([])

    @app.get("/api/watchlist")
    def api_watchlist(request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            return JSONResponse([dict(row) for row in recorder.list_watchlist(user_id=user_id)])
        finally:
            recorder.close()

    @app.post("/api/watchlist")
    def api_watchlist_upsert(item: WatchlistItemIn, request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            recorder.upsert_watchlist(
                item.symbol, item.name, item.sector, item.notes,
                item.sources, item.timeframes, item.profiles, user_id=user_id,
            )
            return JSONResponse({"ok": True})
        finally:
            recorder.close()

    @app.post("/api/analyze")
    def api_analyze(req: AnalyzeRequest, request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        current_cfg = _get_runtime(user_id).current_cfg()
        if len(req.symbols) > current_cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {current_cfg.max_symbols_per_run} sembol seçebilirsiniz.",
            )
        symbols = [{
            "symbol": s.symbol,
            "name": s.name,
            "timeframe": s.timeframe or "1d",
            "profile": s.profile or "balanced",
            "news_sources": s.news_sources or ["google"],
        } for s in req.symbols]
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started})

    @app.post("/api/analyze/multi")
    def api_analyze_multi(req: AnalyzeRequest, request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        current_cfg = _get_runtime(user_id).current_cfg()
        if len(req.symbols) > current_cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {current_cfg.max_symbols_per_run} sembol seçebilirsiniz.",
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
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started, "runs": len(symbols)})

    @app.post("/api/analyze/compare")
    def api_analyze_compare(req: AnalyzeRequest, request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        current_cfg = _get_runtime(user_id).current_cfg()
        if len(req.symbols) > current_cfg.max_symbols_per_run:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"En fazla {current_cfg.max_symbols_per_run} sembol seçebilirsiniz.",
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
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started, "comparisons": len(symbols)})

    @app.get("/api/state")
    def api_state(request: Request, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        state = _get_state(user_id)
        with state.lock:
            return JSONResponse({
                "status": state.status,
                "progress": state.progress,
                "error": state.error,
                "results": state.results,
            })

    @app.get("/api/dashboard")
    def api_dashboard(request: Request, days: int = 30, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            watchlist = [dict(row) for row in recorder.list_watchlist(user_id=user_id)]
            recent = [dict(row) for row in recorder.recent(limit=50, user_id=user_id)]
            hit_hits, hit_total = recorder.hit_rate(days, user_id=user_id)
            by_profile = [dict(row) for row in recorder.summary_by_profile(days, user_id=user_id)]
            by_timeframe = [dict(row) for row in recorder.summary_by_timeframe(days, user_id=user_id)]
            by_direction = [dict(row) for row in recorder.summary_by_direction(days, user_id=user_id)]
            by_symbol = [dict(row) for row in recorder.summary_by_symbol(days, limit=10, user_id=user_id)]
            user = recorder.get_user_by_id(user_id)
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
            "user": {"id": int(user["id"]), "username": str(user["username"])} if user else None,
            "hit_rate": {"hits": hit_hits, "total": hit_total},
            "settings": _get_runtime(user_id).snapshot(),
            "by_profile": by_profile,
            "by_timeframe": by_timeframe,
            "by_direction": by_direction,
            "by_symbol": by_symbol,
            "hit_series": hit_series[-30:],
            "recent": recent,
        })

    @app.get("/api/history")
    def api_history(request: Request, days: int = 7, _: int = Depends(require_user)) -> JSONResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            rows = recorder.recent(limit=50, user_id=user_id)
            hits, total = recorder.hit_rate(days, user_id=user_id)
        finally:
            recorder.close()
        return JSONResponse({
            "hit_rate": {"hits": hits, "total": total},
            "recent": [dict(r) for r in rows],
        })

    @app.get("/api/history.csv")
    def api_history_csv(request: Request, _: int = Depends(require_user)) -> PlainTextResponse:
        user_id = require_user(request)
        recorder = _recorder()
        try:
            rows = recorder.recent(limit=500, user_id=user_id)
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
    def index() -> str:
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
.settings-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:12px;color:var(--muted)}
.field input,.field select{width:100%;background:rgba(255,255,255,.04);color:var(--text);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:13px;outline:none}
.field input:focus,.field select:focus{border-color:#35558b;box-shadow:0 0 0 4px rgba(108,167,255,.12)}
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

    <div class="card panel" id="authCard">
        <div class="row" style="justify-content:space-between;margin-bottom:10px">
            <div>
                <h3 style="margin:0">Hesap</h3>
                <div class="sub">Kendi hesabını oluştur ya da giriş yap. Ayarlar ve watchlist bu hesaba özel kaydedilir.</div>
            </div>
            <div id="meStatus" class="pill">oturum yok</div>
        </div>
        <div class="settings-grid" style="grid-template-columns:repeat(3,1fr)">
            <div class="field"><label>Kullanıcı adı</label><input id="auth_username" type="text" autocomplete="username"></div>
            <div class="field"><label>Şifre</label><input id="auth_password" type="password" autocomplete="current-password"></div>
            <div class="field"><label>&nbsp;</label><div class="row"><button class="btn good" onclick="registerUser()">Kaydol</button><button class="btn alt" onclick="loginUser()">Giriş Yap</button><button class="btn alt" onclick="logoutUser()">Çıkış</button></div></div>
        </div>
        <div id="auth_status" class="sub" style="margin-top:8px"></div>
    </div>

    <div id="app_shell" style="display:none">

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

        <div class="card panel">
            <div class="row" style="justify-content:space-between;margin-bottom:6px">
                <h3 style="margin:0">Uygulama Ayarları</h3>
                <span class="muted">Model, ağırlıklar ve veri ufku burada değişir</span>
            </div>
            <div class="settings-grid">
                <div class="field"><label>Groq model</label><input id="set_groq_model" type="text"></div>
                <div class="field"><label>News weight</label><input id="set_news_weight" type="text"></div>
                <div class="field"><label>Technical weight</label><input id="set_technical_weight" type="text"></div>
                <div class="field"><label>Neutral band</label><input id="set_neutral_band" type="text"></div>
                <div class="field"><label>News lookback hours</label><input id="set_news_lookback_hours" type="text"></div>
                <div class="field"><label>Max articles / symbol</label><input id="set_max_articles_per_symbol" type="text"></div>
                <div class="field"><label>Max symbols / run</label><input id="set_max_symbols_per_run" type="text"></div>
                <div class="field"><label>Intraday lookback</label><input id="set_intraday_lookback_period" type="text"></div>
                <div class="field"><label>Technical lookback</label><input id="set_technical_lookback_period" type="text"></div>
            </div>
            <div class="row" style="margin-top:12px;justify-content:flex-end">
                <button class="btn alt" onclick="loadSettings()">Yenile</button>
                <button class="btn good" onclick="saveSettings()">Ayarları Kaydet</button>
            </div>
            <div id="settings_status" class="sub" style="margin-top:8px"></div>
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
</div>

<script>
let selected = [];
let pollTimer = null;
let hitChart = null;
let barChart = null;
let appSettings = {};

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

function editWatchlist(item){
    const newsSources = (item.sources || 'google').split(',').map(s => s.trim()).filter(Boolean);
    const existing = selected.find(s => s.symbol === item.symbol);
    const payload = {
        symbol: item.symbol,
        name: item.name || item.symbol,
        timeframe: item.timeframes || '1d',
        profile: item.profiles || 'balanced',
        news_sources: newsSources.length ? newsSources : ['google'],
    };
    if (existing){
        Object.assign(existing, payload);
    } else {
        selected.push(payload);
    }
    renderChips();
}

function applySettingsToForm(settings){
    appSettings = settings;
    for (const [key, value] of Object.entries(settings)){
        const input = document.getElementById(`set_${key}`);
        if (input) input.value = value;
    }
}

async function loadSettings(){
    const r = await fetch('/api/settings');
    const s = await r.json();
    applySettingsToForm(s);
    document.getElementById('settings_status').textContent = 'Ayarlar yüklendi.';
}

async function saveSettings(){
    const fallback = (key) => appSettings[key];
    const numberOrFallback = (value, key) => {
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : fallback(key);
    };
    const payload = {
        groq_model: document.getElementById('set_groq_model').value.trim() || fallback('groq_model'),
        news_weight: numberOrFallback(document.getElementById('set_news_weight').value, 'news_weight'),
        technical_weight: numberOrFallback(document.getElementById('set_technical_weight').value, 'technical_weight'),
        neutral_band: numberOrFallback(document.getElementById('set_neutral_band').value, 'neutral_band'),
        news_lookback_hours: numberOrFallback(document.getElementById('set_news_lookback_hours').value, 'news_lookback_hours'),
        max_articles_per_symbol: numberOrFallback(document.getElementById('set_max_articles_per_symbol').value, 'max_articles_per_symbol'),
        max_symbols_per_run: numberOrFallback(document.getElementById('set_max_symbols_per_run').value, 'max_symbols_per_run'),
        intraday_lookback_period: document.getElementById('set_intraday_lookback_period').value.trim() || fallback('intraday_lookback_period'),
        technical_lookback_period: document.getElementById('set_technical_lookback_period').value.trim() || fallback('technical_lookback_period'),
    };
    const r = await fetch('/api/settings', {
        method:'PUT', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload),
    });
    if (!r.ok){
        const err = await r.json().catch(() => ({}));
        document.getElementById('settings_status').textContent = err.detail || 'Ayarlar kaydedilemedi.';
        return;
    }
    const saved = await r.json();
    applySettingsToForm(saved);
    document.getElementById('settings_status').textContent = 'Ayarlar kaydedildi ve aktif edildi.';
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

    document.getElementById('watchlist').innerHTML = d.watchlist.length ? d.watchlist.map(w => `<tr><td><b>${w.symbol}</b></td><td>${w.name || ''}</td><td>${w.profiles || ''} / ${w.timeframes || ''}</td><td class="muted">${w.sources || ''} <button class="btn alt" style="padding:6px 10px;margin-left:8px" onclick='editWatchlist(${JSON.stringify(w).replace(/'/g,"&#39;")})'>Seç</button></td></tr>`).join('') : '<tr><td colspan="4" class="empty">Watchlist boş</td></tr>';
}

async function loadWatchlist(){
  const r = await fetch('/api/watchlist');
  const items = await r.json();
  if (!items.length) return;
    document.getElementById('watchlist').innerHTML = items.map(w => `<tr><td><b>${w.symbol}</b></td><td>${w.name || ''}</td><td>${w.profiles || ''} / ${w.timeframes || ''}</td><td class="muted">${w.sources || ''} <button class="btn alt" style="padding:6px 10px;margin-left:8px" onclick='editWatchlist(${JSON.stringify(w).replace(/'/g,"&#39;")})'>Seç</button></td></tr>`).join('');
}

async function loginUser(){
    const payload = {username: document.getElementById('auth_username').value, password: document.getElementById('auth_password').value};
    const r = await fetch('/api/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    if (!r.ok){
        const err = await r.json().catch(() => ({}));
        document.getElementById('auth_status').textContent = err.detail || 'Giriş başarısız.';
        return;
    }
    document.getElementById('auth_status').textContent = 'Giriş başarılı.';
    await initializeForSession();
}

async function registerUser(){
    const payload = {username: document.getElementById('auth_username').value, password: document.getElementById('auth_password').value};
    const r = await fetch('/api/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    if (!r.ok){
        const err = await r.json().catch(() => ({}));
        document.getElementById('auth_status').textContent = err.detail || 'Kayıt başarısız.';
        return;
    }
    document.getElementById('auth_status').textContent = 'Hesap oluşturuldu.';
    await initializeForSession();
}

async function logoutUser(){
    await fetch('/api/logout', {method:'POST'});
    selected = [];
    renderChips();
    document.getElementById('app_shell').style.display = 'none';
    document.getElementById('meStatus').textContent = 'oturum yok';
    document.getElementById('auth_status').textContent = 'Çıkış yapıldı.';
}

async function loadMe(){
    const r = await fetch('/api/me');
    if (!r.ok) return null;
    return await r.json();
}

async function initializeForSession(){
    const me = await loadMe();
    if (!me){
        document.getElementById('app_shell').style.display = 'none';
        document.getElementById('meStatus').textContent = 'oturum yok';
        return;
    }
    document.getElementById('meStatus').textContent = `giriş: ${me.username}`;
    document.getElementById('app_shell').style.display = 'block';
    await loadSettings();
    await loadWatchlist();
    await loadHistory();
    await loadDashboard();
}

initializeForSession();
</script></body></html>"""