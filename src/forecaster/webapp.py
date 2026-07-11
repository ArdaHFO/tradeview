"""Web UI: pick symbols, trigger analysis on demand, see results (FastAPI, single page)."""
from __future__ import annotations

import csv
import io
import logging
import re
import secrets
import threading
import time
from dataclasses import replace

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from .config import Config
from .models import Prediction
from .news.fetch import fetch_articles
from .pipeline import load_watchlist, run_for_symbols
from .storage.recorder import PredictionRecorder
from .symbols_search import search_symbols
from .technical.data import ALLOWED_TIMEFRAMES, fetch_bars
from .technical.indicators import bollinger_bands, ema, sma

log = logging.getLogger(__name__)
COOKIE_NAME = "tradeview_session"


def _parse_timeframes(raw: str | None) -> list[str]:
    """Split a comma-separated timeframe string and drop anything not in
    ALLOWED_TIMEFRAMES — never let a raw client string reach yfinance as an
    interval (a single "1d,1wk,1mo" string is not a valid yfinance interval).
    """
    if not raw:
        return ["1d"]
    valid = [tf.strip() for tf in raw.split(",") if tf.strip() in ALLOWED_TIMEFRAMES]
    # dedupe while preserving order
    seen: set[str] = set()
    out = [tf for tf in valid if not (tf in seen or seen.add(tf))]
    return out or ["1d"]


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
    invite_code: str | None = None


class _RateLimiter:
    """In-memory sliding-window limiter keyed by client IP.

    Single-process only, consistent with this app's other in-memory state
    (AnalysisState/RuntimeState caches) — see the multi-worker note in
    create_app.
    """

    def __init__(self, max_attempts: int, window_seconds: float) -> None:
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            attempts = [t for t in self._attempts.get(key, []) if now - t < self.window_seconds]
            attempts.append(now)
            self._attempts[key] = attempts
            return len(attempts) <= self.max_attempts


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class WatchlistItemIn(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    notes: str | None = None
    sources: str = "google"
    timeframes: str = "1d"
    profiles: str = "balanced"


_PERIOD_RE = re.compile(r"^(\d+(d|mo|y)|ytd|max)$")


class AppSettingsIn(BaseModel):
    news_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    technical_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    neutral_band: float | None = Field(default=None, ge=0.0, le=1.0)
    groq_model: str | None = Field(default=None, min_length=1)
    news_lookback_hours: int | None = Field(default=None, ge=1, le=168)
    max_articles_per_symbol: int | None = Field(default=None, ge=1, le=50)
    max_symbols_per_run: int | None = Field(default=None, ge=1, le=100)
    intraday_lookback_period: str | None = None
    technical_lookback_period: str | None = None

    @field_validator("intraday_lookback_period", "technical_lookback_period")
    @classmethod
    def _validate_period(cls, value: str | None) -> str | None:
        # These feed straight into yfinance's `period=` argument — an
        # unvalidated string here is the same class of bug as the timeframe
        # string that used to reach the `interval=` argument unvalidated.
        if value is not None and not _PERIOD_RE.match(value):
            raise ValueError(
                "period must look like '60d', '6mo', '5y', 'ytd', or 'max'")
        return value


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
        "technical_indicators": [
            {
                "name": ind.name, "value": ind.value, "direction": ind.direction.value,
                "weight_pct": ind.weight_pct, "explanation": ind.explanation,
            }
            for ind in p.technical_indicators
        ],
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
    # 8 attempts / 5 minutes per IP on register+login combined — generous
    # enough for typo-driven retries, tight enough to blunt brute force.
    auth_limiter = _RateLimiter(max_attempts=8, window_seconds=300)

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
    def api_register(item: AuthRequest, request: Request, response: Response) -> JSONResponse:
        if not auth_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                 detail="Too many attempts, try again later")
        username = item.username.strip().lower()
        if not username or not item.password:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username and password are required")
        if cfg.registration_code and item.invite_code != cfg.registration_code:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid registration code")
        recorder = _recorder()
        try:
            if recorder.get_user(username) is not None:
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="User already exists")
            user_id = recorder.create_user(username, item.password)
            token = recorder.create_session(user_id)
        finally:
            recorder.close()
        response = JSONResponse({"ok": True, "username": username})
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax",
                             secure=cfg.cookie_secure, max_age=60 * 60 * 24 * 7)
        return response

    @app.post("/api/login")
    def api_login(item: AuthRequest, request: Request) -> JSONResponse:
        if not auth_limiter.allow(_client_ip(request)):
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                                 detail="Too many attempts, try again later")
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
        response.set_cookie(COOKIE_NAME, token, httponly=True, samesite="lax",
                             secure=cfg.cookie_secure, max_age=60 * 60 * 24 * 7)
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

    @app.get("/api/news")
    def api_news(symbol: str, name: str = "", sources: str = "google",
                 user_id: int = Depends(require_user)) -> JSONResponse:
        """Raw article list for a symbol — no AI scoring, no Groq cost."""
        current_cfg = _get_runtime(user_id).current_cfg()
        source_list = [s.strip() for s in sources.split(",") if s.strip()]
        articles = fetch_articles(symbol, name or None, current_cfg, source_list)
        return JSONResponse([
            {
                "title": a.title, "source": a.source, "url": a.url,
                "published_ts": a.published_ts.isoformat(), "snippet": a.snippet,
            }
            for a in articles
        ])

    @app.get("/api/chart")
    def api_chart(symbol: str, timeframe: str = "1d",
                   user_id: int = Depends(require_user)) -> JSONResponse:
        """Price series + indicator overlays for charting — no Groq cost."""
        if timeframe not in ALLOWED_TIMEFRAMES:
            timeframe = "1d"
        current_cfg = _get_runtime(user_id).current_cfg()
        bars = fetch_bars(symbol, current_cfg, timeframe)
        closes = [b.close for b in bars]
        sma50 = sma(closes, 50)
        sma200 = sma(closes, 200) if len(closes) >= 200 else [None] * len(closes)
        ema20 = ema(closes, 20)
        upper, mid, lower = bollinger_bands(closes)
        return JSONResponse({
            "symbol": symbol,
            "dates": [b.ts.isoformat() for b in bars],
            "close": closes,
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
        })

    @app.get("/api/settings")
    def api_settings(user_id: int = Depends(require_user)) -> JSONResponse:
        return JSONResponse(_get_runtime(user_id).snapshot())

    @app.put("/api/settings")
    def api_settings_update(item: AppSettingsIn, user_id: int = Depends(require_user)) -> JSONResponse:
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
    def api_watchlist(user_id: int = Depends(require_user)) -> JSONResponse:
        recorder = _recorder()
        try:
            return JSONResponse([dict(row) for row in recorder.list_watchlist(user_id=user_id)])
        finally:
            recorder.close()

    @app.post("/api/watchlist")
    def api_watchlist_upsert(item: WatchlistItemIn, user_id: int = Depends(require_user)) -> JSONResponse:
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
    def api_analyze(req: AnalyzeRequest, user_id: int = Depends(require_user)) -> JSONResponse:
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
            for timeframe in _parse_timeframes(s.timeframe)
        ]
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started, "runs": len(symbols)})

    @app.post("/api/analyze/multi")
    def api_analyze_multi(req: AnalyzeRequest, user_id: int = Depends(require_user)) -> JSONResponse:
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
            for timeframe in _parse_timeframes(s.timeframe)
        ]
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started, "runs": len(symbols)})

    @app.post("/api/analyze/compare")
    def api_analyze_compare(req: AnalyzeRequest, user_id: int = Depends(require_user)) -> JSONResponse:
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
                "timeframe": timeframe,
                "profile": profile,
                "news_sources": s.news_sources or ["google"],
            }
            for s in req.symbols
            for timeframe in _parse_timeframes(s.timeframe)
            for profile in profiles
        ]
        started = _get_state(user_id).start_analysis(symbols, user_id=user_id)
        return JSONResponse({"started": started, "comparisons": len(symbols)})

    @app.get("/api/state")
    def api_state(user_id: int = Depends(require_user)) -> JSONResponse:
        state = _get_state(user_id)
        with state.lock:
            return JSONResponse({
                "status": state.status,
                "progress": state.progress,
                "error": state.error,
                "results": state.results,
            })

    @app.get("/api/dashboard")
    def api_dashboard(days: int = 30, user_id: int = Depends(require_user)) -> JSONResponse:
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

        # Only resolved predictions belong in a running hit-rate — a pending
        # row (hit IS NULL) is not yet a miss, and counting it as one (via
        # `1 if row.get("hit") else 0`, which is also 0 for None) understated
        # the rate for every symbol with an in-flight prediction.
        resolved = [row for row in recent if row.get("hit") is not None]
        hit_series = []
        running_hits = 0
        running_total = 0
        for row in reversed(resolved):
            running_total += 1
            running_hits += 1 if row["hit"] else 0
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
    def api_history(days: int = 7, user_id: int = Depends(require_user)) -> JSONResponse:
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
    def api_history_csv(user_id: int = Depends(require_user)) -> PlainTextResponse:
        recorder = _recorder()
        try:
            rows = recorder.recent(limit=500, user_id=user_id)
        finally:
            recorder.close()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["ts", "symbol", "timeframe", "profile", "final_score", "final_direction",
                          "final_confidence", "actual_direction", "hit"])
        for r in rows:
            writer.writerow([r["ts"], r["symbol"], r["timeframe"], r["profile"], r["final_score"],
                             r["final_direction"], r["final_confidence"], r["actual_direction"], r["hit"]])
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
<title>TradeView — Küresel Haber + Teknik Analiz</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
:root{--bg:#07111f;--bg2:#0c1629;--panel:#101b30;--panel2:#13213a;--text:#e7eefb;--muted:#8ea4c7;--line:#223557;--green:#31c48d;--red:#ff6b6b;--amber:#f4b942;--blue:#6ca7ff}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;font:14px/1.55 Inter,Segoe UI,sans-serif;color:var(--text);background:radial-gradient(circle at top left, rgba(108,167,255,.18), transparent 28%),radial-gradient(circle at top right, rgba(49,196,141,.12), transparent 22%),linear-gradient(180deg,var(--bg),var(--bg2))}
.wrap{max-width:1280px;margin:0 auto;padding:28px 20px 40px}
.hero{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;flex-wrap:wrap;margin-bottom:18px}
h1{margin:0;font-size:30px;letter-spacing:-.02em}
h1 .tag{font-size:13px;font-weight:600;color:var(--muted);vertical-align:middle;margin-left:8px}
.sub{color:var(--muted);margin-top:6px}
.badge{display:inline-flex;align-items:center;gap:8px;padding:6px 12px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.03)}
.pill{display:inline-flex;align-items:center;padding:2px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--line);color:var(--muted);font-size:12px}
.pill.good{color:var(--green);border-color:rgba(49,196,141,.4)}.pill.warn{color:var(--amber);border-color:rgba(244,185,66,.4)}.pill.danger{color:var(--red);border-color:rgba(255,107,107,.4)}
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
.btn.small{padding:6px 10px;font-size:12px}
.settings-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:10px}
.field{display:flex;flex-direction:column;gap:6px}
.field label{font-size:12px;color:var(--muted)}
.field input,.field select{width:100%;background:rgba(255,255,255,.04);color:var(--text);border:1px solid var(--line);border-radius:12px;padding:10px 12px;font-size:13px;outline:none}
.field input:focus,.field select:focus{border-color:#35558b;box-shadow:0 0 0 4px rgba(108,167,255,.12)}
table{width:100%;border-collapse:collapse}th,td{padding:11px 10px;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top;text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}td{color:#dce7f8}
.up{color:var(--green);font-weight:700}.down{color:var(--red);font-weight:700}.neutral{color:var(--muted);font-weight:700}.reasons,.muted{color:var(--muted)}.empty{color:var(--muted);padding:18px 10px;font-style:italic;text-align:center}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:999px;display:inline-block}
canvas{width:100%!important;height:320px!important}
.row-clickable{cursor:pointer}
.row-clickable:hover td{background:rgba(255,255,255,.03)}
.skel{background:linear-gradient(90deg, rgba(255,255,255,.04) 25%, rgba(255,255,255,.09) 37%, rgba(255,255,255,.04) 63%);background-size:400% 100%;animation:skel 1.4s ease infinite;border-radius:8px;height:14px}
@keyframes skel{0%{background-position:100% 50%}100%{background-position:0 50%}}
.modal-overlay{position:fixed;inset:0;background:rgba(3,8,16,.72);display:none;align-items:flex-start;justify-content:center;z-index:100;padding:32px 16px;overflow-y:auto}
.modal-overlay.open{display:flex}
.modal{background:var(--panel);border:1px solid var(--line);border-radius:18px;max-width:960px;width:100%;padding:22px;box-shadow:0 30px 80px rgba(0,0,0,.5)}
.modal-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;gap:12px}
.modal-close{background:none;border:1px solid var(--line);color:var(--muted);border-radius:10px;padding:6px 14px;cursor:pointer;font-size:14px}
.modal-close:hover{background:rgba(255,255,255,.06)}
.score-badges{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.score-badge{display:flex;flex-direction:column;gap:4px;padding:10px 14px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid var(--line);min-width:120px}
.score-badge .num{font-size:20px;font-weight:800}
.score-bar{height:6px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:4px}
.score-bar i{display:block;height:100%;border-radius:999px}
.ind-table td.dirUP{color:var(--green);font-weight:700}.ind-table td.dirDOWN{color:var(--red);font-weight:700}.ind-table td.dirNEUTRAL{color:var(--muted);font-weight:700}
.news-item{padding:10px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.news-item a{color:var(--text);text-decoration:none;font-weight:600}
.news-item a:hover{text-decoration:underline;color:var(--blue)}
.news-meta{color:var(--muted);font-size:12px;margin-top:2px}
.section-title{margin:20px 0 10px;font-size:15px;font-weight:700;display:flex;align-items:center;gap:8px}
@media (max-width:1000px){.stat,.half{grid-column:span 12}.settings-grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body>
<div class="wrap">
  <div class="hero">
    <div>
      <h1>TradeView <span class="tag">🌍 Türkiye · ABD · Avrupa · Asya — tüm dünya piyasaları</span></h1>
      <div class="sub">Haber + teknik analiz, yapay zekâ destekli yorumlama, favori listesi ve performans panosu.</div>
    </div>
    <div class="badge"><span id="modelBadge" class="pill">AI Modeli: —</span><span id="status" class="pill">Boşta</span></div>
  </div>

    <div class="card panel" id="authCard">
        <div class="row" style="justify-content:space-between;margin-bottom:10px">
            <div>
                <h3 style="margin:0">Hesap</h3>
                <div class="sub">Kendi hesabını oluştur ya da giriş yap. Ayarlar ve favori listen bu hesaba özel kaydedilir.</div>
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
    <div class="card stat"><div class="l">Son 30 Gün İsabet Oranı</div><div id="statHit" class="v">-</div></div>
    <div class="card stat"><div class="l">Favori Sayısı</div><div id="statWatchlist" class="v">-</div></div>
    <div class="card stat"><div class="l">Son Kullanılan Profil</div><div id="statModel" class="v">-</div></div>
    <div class="card stat"><div class="l">Son Kullanılan Zaman Dilimi</div><div id="statTf" class="v">-</div></div>

    <div class="card search">
      <div class="row" style="justify-content:space-between;margin-bottom:10px">
        <div class="muted">Dünyanın herhangi bir borsasından sembol veya şirket adı ara (ör. AAPL, ASELS, THYAO, SAP, MC), seç ve analiz et.</div>
        <div class="legend"><span><i class="dot" style="background:var(--green)"></i> Yükseliş</span><span><i class="dot" style="background:var(--red)"></i> Düşüş</span><span><i class="dot" style="background:var(--muted)"></i> Nötr</span></div>
      </div>
      <input type="text" id="q" placeholder="AAPL, ASELS, THYAO, SAP, MC, Apple, Tesla..." autocomplete="off">
      <div id="dd" class="dropdown" style="display:none"></div>
      <div class="row" style="margin-top:12px;gap:16px;flex-wrap:wrap">
        <div class="field" style="min-width:220px">
          <label>Zaman dilimi (yeni eklenecek semboller için)</label>
          <div class="row" id="tf_controls">
            <label><input type="checkbox" class="tf_cb" value="1d" checked> 1 gün</label>
            <label><input type="checkbox" class="tf_cb" value="1h"> 1 saat</label>
            <label><input type="checkbox" class="tf_cb" value="30m"> 30 dk</label>
            <label><input type="checkbox" class="tf_cb" value="1wk"> 1 hafta</label>
            <label><input type="checkbox" class="tf_cb" value="1mo"> 1 ay</label>
          </div>
        </div>
        <div class="field" style="min-width:180px">
          <label>Analiz profili</label>
          <select id="profile_control">
            <option value="balanced">Dengeli (haber + teknik)</option>
            <option value="news_heavy">Haber ağırlıklı</option>
            <option value="technical_heavy">Teknik ağırlıklı</option>
            <option value="news_only">Sadece haber</option>
            <option value="technical_only">Sadece teknik</option>
          </select>
        </div>
        <div class="field" style="min-width:160px">
          <label>Haber kaynağı</label>
          <div class="row" id="src_controls">
            <label><input type="checkbox" class="src_cb" value="google" checked> Google Haberler</label>
            <label><input type="checkbox" class="src_cb" value="yahoo"> Yahoo Finans</label>
          </div>
        </div>
      </div>
      <div class="row" style="margin-top:12px;justify-content:space-between">
        <div class="chips" id="chips"></div>
        <div class="row">
          <button class="btn good" id="go" onclick="analyze()">▶ Analiz Et</button>
          <button class="btn alt" onclick="compareModels()">⚖ Profil Karşılaştır</button>
          <button class="btn alt" onclick="multiTimeframe()">⏱ Çok Zaman Dilimi</button>
          <button class="btn alt" onclick="saveWatchlist()">★ Favorilere Ekle</button>
          <button class="btn alt" onclick="loadDashboard()">↻ Panoyu Yenile</button>
        </div>
      </div>
      <div id="progress" class="sub" style="margin-top:10px"></div>
    </div>

        <div class="card panel">
            <div class="row" style="justify-content:space-between;margin-bottom:6px">
                <h3 style="margin:0">Uygulama Ayarları</h3>
                <span class="muted">Yapay zekâ modeli, ağırlıklar ve veri ufku burada değişir</span>
            </div>
            <div class="settings-grid">
                <div class="field"><label>AI Modeli</label><input id="set_groq_model" type="text"></div>
                <div class="field"><label>Haber Ağırlığı (0-1)</label><input id="set_news_weight" type="text"></div>
                <div class="field"><label>Teknik Ağırlık (0-1)</label><input id="set_technical_weight" type="text"></div>
                <div class="field"><label>Nötr Bant (0-1)</label><input id="set_neutral_band" type="text"></div>
                <div class="field"><label>Haber Geriye Bakış (saat)</label><input id="set_news_lookback_hours" type="text"></div>
                <div class="field"><label>Sembol Başına Maks. Haber</label><input id="set_max_articles_per_symbol" type="text"></div>
                <div class="field"><label>Çalıştırma Başına Maks. Sembol</label><input id="set_max_symbols_per_run" type="text"></div>
                <div class="field"><label>Gün İçi Veri Aralığı</label><input id="set_intraday_lookback_period" type="text"></div>
                <div class="field"><label>Teknik Veri Aralığı</label><input id="set_technical_lookback_period" type="text"></div>
            </div>
            <div class="row" style="margin-top:12px;justify-content:flex-end">
                <button class="btn alt" onclick="loadSettings()">Yenile</button>
                <button class="btn good" onclick="saveSettings()">Ayarları Kaydet</button>
            </div>
            <div id="settings_status" class="sub" style="margin-top:8px"></div>
        </div>

    <div class="card half"><h3 style="margin:0 0 10px">📈 Performans (Kümülatif İsabet Oranı)</h3><canvas id="hitChart"></canvas></div>
    <div class="card half"><h3 style="margin:0 0 10px">📊 Profil / Zaman Dilimi Bazında İsabet</h3><canvas id="barChart"></canvas></div>

    <div class="card panel" id="compareCard" style="display:none">
      <h3 style="margin:0 0 4px">⚖ Hisse Karşılaştırma</h3>
      <div class="sub" style="margin-bottom:10px">Bu çalıştırmada analiz edilen semboller yan yana.</div>
      <canvas id="compareChartCanvas" style="margin-bottom:14px"></canvas>
      <table><thead><tr><th>Sembol</th><th>Yön</th><th>Final Skor</th><th>Haber Skoru</th><th>Teknik Skor</th><th>Güven</th></tr></thead>
      <tbody id="compareTable"></tbody></table>
    </div>

    <div class="card panel"><h3 style="margin:0 0 10px">🔍 Son Analiz Sonuçları <span class="muted" style="font-weight:400;font-size:12px">— bir satıra tıklayarak haberleri, grafiği ve gösterge detaylarını gör</span></h3>
      <table><thead><tr><th>Sembol</th><th>Zaman Dilimi</th><th>Profil</th><th>Yön</th><th>Final</th><th>Güven</th><th>Haber</th><th>Teknik</th><th>Detay</th></tr></thead><tbody id="results"><tr><td colspan="9" class="empty">Henüz analiz yok — yukarıdan bir sembol seçip "Analiz Et" butonuna basın.</td></tr></tbody></table>
    </div>

    <div class="card half"><h3 style="margin:0 0 10px">★ Favori Listem</h3><table><thead><tr><th>Sembol</th><th>Ad</th><th>Profil / Zaman Dilimi</th><th>Kaynak</th></tr></thead><tbody id="watchlist"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table></div>
    <div class="card half"><h3 style="margin:0 0 10px">🕓 Geçmiş Tahminler</h3><table><thead><tr><th>Zaman</th><th>Sembol</th><th>Yön</th><th>İsabet</th></tr></thead><tbody id="history"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table></div>
    </div>

    </div>
</div>

<div class="modal-overlay" id="detailModal" onclick="if(event.target===this) closeDetail()">
  <div class="modal">
    <div class="modal-header">
      <div>
        <h2 id="detailTitle" style="margin:0">—</h2>
        <div id="detailSub" class="sub"></div>
      </div>
      <button class="modal-close" onclick="closeDetail()">✕ Kapat</button>
    </div>
    <div class="score-badges" id="detailScores"></div>

    <div class="section-title">💬 Yapay Zekâ Haber Yorumu</div>
    <div id="detailNewsRationale" class="muted"></div>

    <div class="section-title">📰 İlgili Haberler</div>
    <div id="detailNews"><div class="muted">Yükleniyor...</div></div>

    <div class="section-title">📈 Fiyat Grafiği (gösterge overlay'leri ile)</div>
    <canvas id="detailChartCanvas"></canvas>

    <div class="section-title">🧮 Teknik Gösterge Detayı</div>
    <div id="detailIndicators"></div>
  </div>
</div>

<script>
let selected = [];
let pollTimer = null;
let hitChart = null;
let barChart = null;
let appSettings = {};

function dirClass(d){ return d==='UP' ? 'up' : (d==='DOWN' ? 'down' : 'neutral'); }
function dirLabel(d){ return d==='UP' ? 'YÜKSELİŞ' : (d==='DOWN' ? 'DÜŞÜŞ' : 'NÖTR'); }
function dirArrow(d){ return d==='UP' ? '▲' : (d==='DOWN' ? '▼' : '–'); }
function statusLabel(s){ return {idle:'Boşta', running:'Çalışıyor', done:'Tamam', error:'Hata'}[s] || s; }
function esc(s){
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function errDetail(err){
  const d = err && err.detail;
  if (!d) return null;
  if (typeof d === 'string') return d;
  if (Array.isArray(d)) return d.map(e => e.msg || JSON.stringify(e)).join('; ');
  return JSON.stringify(d);
}
function pct(hit, total){ return total ? Math.round(100 * hit / total) + '%' : '0%'; }
function currencyForSymbol(sym){
  const suf = (sym || '').includes('.') ? sym.split('.').pop().toUpperCase() : '';
  const map = {IS:'₺', PA:'€', DE:'€', F:'€', AS:'€', MI:'€', MC:'€', L:'£', SW:'CHF', T:'¥', HK:'HK$', KS:'₩', SA:'R$'};
  return map[suf] || '$';
}
function formatPrice(sym, value){
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return currencyForSymbol(sym) + Number(value).toFixed(2);
}

const qEl = document.getElementById('q');
let searchDebounce = null;
let lastSearchResults = [];

async function runSearch(){
  const q = qEl.value.trim();
  const dd = document.getElementById('dd');
  if (q.length < 2) { dd.style.display = 'none'; lastSearchResults = []; return; }
  const r = await fetch('/api/symbols?q=' + encodeURIComponent(q));
  const items = await r.json();
  lastSearchResults = items;
  if (!items.length) { dd.style.display = 'none'; return; }
  dd.innerHTML = items.map(it => `<div onclick='pick(${JSON.stringify(it).replace(/'/g,"&#39;")})'><b>${esc(it.symbol)}</b> — ${esc(it.name)}</div>`).join('');
  dd.style.display = 'block';
}

qEl.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(runSearch, 300);
});

qEl.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && lastSearchResults.length){
    ev.preventDefault();
    pick(lastSearchResults[0]);
  } else if (ev.key === 'Escape'){
    document.getElementById('dd').style.display = 'none';
  }
});

document.addEventListener('click', (ev) => {
  const dd = document.getElementById('dd');
  if (ev.target !== qEl && !dd.contains(ev.target)) dd.style.display = 'none';
});

function checkedValues(selector){
  return [...document.querySelectorAll(selector + ':checked')].map(el => el.value);
}
function controlTimeframes(){ return checkedValues('.tf_cb').join(',') || '1d'; }
function controlSources(){ return checkedValues('.src_cb'); }
function controlProfile(){ return document.getElementById('profile_control').value || 'balanced'; }

function pick(item){
  if (!selected.some(s => s.symbol === item.symbol)) {
    selected.push({
      ...item,
      timeframe: controlTimeframes(),
      profile: controlProfile(),
      news_sources: controlSources().length ? controlSources() : ['google'],
    });
  }
  qEl.value = '';
  document.getElementById('dd').style.display = 'none';
  renderChips();
}

function remove(symbol){ selected = selected.filter(s => s.symbol !== symbol); renderChips(); }
function renderChips(){
  document.getElementById('chips').innerHTML = selected.map(s =>
    `<span class="chip" title="${esc(s.timeframe)} / ${esc(s.profile)} / ${esc((s.news_sources||[]).join(','))}">${esc(s.symbol)}
      <button onclick="remove('${esc(s.symbol)}')">×</button></span>`).join('');
}

async function postAnalyze(url, payload){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (!r.ok){
    const err = await r.json().catch(() => ({}));
    alert(errDetail(err) || 'İşlem başlatılamadı.');
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
    document.getElementById('modelBadge').textContent = 'AI Modeli: ' + esc(settings.groq_model || '—');
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
        document.getElementById('settings_status').textContent = errDetail(err) || 'Ayarlar kaydedilemedi.';
        return;
    }
    const saved = await r.json();
    applySettingsToForm(saved);
    document.getElementById('settings_status').textContent = 'Ayarlar kaydedildi ve aktif edildi.';
    loadDashboard();
}

function startPolling(){ if (pollTimer) return; pollTimer = setInterval(refreshState, 1500); refreshState(); }

let lastResults = [];

function skeletonRows(cols, rows){
  const cells = Array.from({length: cols}, () => `<td><div class="skel"></div></td>`).join('');
  return Array.from({length: rows}, () => `<tr>${cells}</tr>`).join('');
}

async function refreshState(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const st = document.getElementById('status');
  st.textContent = statusLabel(s.status) + (s.progress ? ' · ' + s.progress : '');
  st.className = 'pill ' + (s.status === 'error' ? 'danger' : (s.status === 'running' ? 'warn' : 'good'));
  document.getElementById('go').disabled = (s.status === 'running');
  document.getElementById('progress').textContent = s.error ? ('Hata: ' + s.error) : (s.progress || '');

  const results = document.getElementById('results');
  if (s.status === 'running' && !s.results.length){
    results.innerHTML = skeletonRows(9, 3);
  } else if (s.results.length){
    lastResults = s.results;
    results.innerHTML = s.results.map((p, i) => `<tr class="row-clickable" onclick="openDetail(${i})">
      <td><b>${esc(p.symbol)}</b></td>
      <td>${esc(p.timeframe)}</td>
      <td>${esc(p.profile)}</td>
      <td class="${dirClass(p.final_direction)}">${dirArrow(p.final_direction)} ${dirLabel(p.final_direction)}</td>
      <td>${p.final_score.toFixed(2)}</td>
      <td>${p.final_confidence.toFixed(2)}</td>
      <td>${p.news_score.toFixed(2)} (${p.news_confidence.toFixed(2)})</td>
      <td>${p.technical_score.toFixed(2)}</td>
      <td><button class="btn alt small" onclick="event.stopPropagation(); openDetail(${i})">🔍 İncele</button></td>
      </tr>`).join('');
    renderComparison(s.results);
  }
  if (s.status !== 'running' && pollTimer){ clearInterval(pollTimer); pollTimer = null; loadHistory(); loadDashboard(); }
}

async function loadHistory(){
  // statHit ("Son 30 gün hit rate") is exclusively loadDashboard's job —
  // this used to also set it from a 7-day window under the same label,
  // and whichever of the two async calls finished last silently won.
  const r = await fetch('/api/history?days=7');
  const s = await r.json();
  document.getElementById('history').innerHTML = s.recent.length ? s.recent.map(p => `<tr>
    <td>${new Date(p.ts).toLocaleString('tr-TR')}</td>
    <td><b>${esc(p.symbol)}</b></td>
    <td class="${dirClass(p.final_direction)}">${dirLabel(p.final_direction)}</td>
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
    data:{ labels:d.hit_series.map(x => new Date(x.ts).toLocaleDateString('tr-TR')), datasets:[{label:'Kümülatif İsabet Oranı (%)', data:d.hit_series.map(x => x.running_hit_rate), borderColor:'#6ca7ff', backgroundColor:'rgba(108,167,255,.18)', tension:.3, fill:true }]},
    options:{ responsive:true, plugins:{ legend:{display:false} }, scales:{ y:{ beginAtZero:true, max:100, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
  });

  // Hit *rate* (%), not raw hit *count* — a profile run 40 times isn't
  // "better" than one run 5 times just because it has more hits.
  const rate = (x) => x.total > 0 ? Math.round(100 * x.hits / x.total) : 0;
  const profileMap = new Map((d.by_profile || []).map(x => [x.profile, rate(x)]));
  const timeframeMap = new Map((d.by_timeframe || []).map(x => [x.timeframe, rate(x)]));
  const labels = [...new Set([...profileMap.keys(), ...timeframeMap.keys()])].filter(Boolean);
  barChart = chartOrUpdate(barChart, document.getElementById('barChart'), {
    type:'bar',
    data:{
      labels,
      datasets:[
        {label:'Profil isabet %', data:labels.map(l => profileMap.has(l) ? profileMap.get(l) : null), backgroundColor:'#31c48d'},
        {label:'Zaman dilimi isabet %', data:labels.map(l => timeframeMap.has(l) ? timeframeMap.get(l) : null), backgroundColor:'#6ca7ff'},
      ],
    },
    options:{ responsive:true, plugins:{ legend:{ labels:{ color:'#e7eefb' } } },
      scales:{ y:{ beginAtZero:true, max:100, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
  });

    document.getElementById('watchlist').innerHTML = d.watchlist.length ? d.watchlist.map(w => `<tr><td><b>${esc(w.symbol)}</b></td><td>${esc(w.name || '')}</td><td>${esc(w.profiles || '')} / ${esc(w.timeframes || '')}</td><td class="muted">${esc(w.sources || '')} <button class="btn alt small" style="margin-left:8px" onclick='editWatchlist(${JSON.stringify(w).replace(/'/g,"&#39;")})'>Seç</button></td></tr>`).join('') : '<tr><td colspan="4" class="empty">Favori listesi boş — bir sembol seçip "Favorilere Ekle" butonuna basın.</td></tr>';
}

let compareChart = null;
function renderComparison(results){
  const card = document.getElementById('compareCard');
  if (!results || results.length < 2){ card.style.display = 'none'; return; }
  card.style.display = '';
  compareChart = chartOrUpdate(compareChart, document.getElementById('compareChartCanvas'), {
    type: 'bar',
    data: {
      labels: results.map(p => p.symbol),
      datasets: [{
        label: 'Final Skor',
        data: results.map(p => p.final_score),
        backgroundColor: results.map(p => p.final_score > 0.15 ? '#31c48d' : (p.final_score < -0.15 ? '#ff6b6b' : '#8ea4c7')),
      }],
    },
    options: { responsive:true, plugins:{legend:{display:false}},
      scales:{ y:{min:-1, max:1, grid:{color:'rgba(255,255,255,.06)'}}, x:{grid:{display:false}} } },
  });
  document.getElementById('compareTable').innerHTML = results.map(p => `<tr>
    <td><b>${esc(p.symbol)}</b></td>
    <td class="${dirClass(p.final_direction)}">${dirArrow(p.final_direction)} ${dirLabel(p.final_direction)}</td>
    <td>${p.final_score.toFixed(2)}</td>
    <td>${p.news_score.toFixed(2)}</td>
    <td>${p.technical_score.toFixed(2)}</td>
    <td>${Math.round(p.final_confidence * 100)}%</td>
    </tr>`).join('');
}

function scoreBadge(label, score, confidence){
  const clamped = Math.max(-1, Math.min(1, score));
  const barPct = Math.round(((clamped + 1) / 2) * 100);
  const color = score > 0.15 ? 'var(--green)' : (score < -0.15 ? 'var(--red)' : 'var(--muted)');
  const confRow = confidence === undefined || confidence === null ? '' :
    `<div class="muted" style="font-size:11px">güven ${Math.round(confidence * 100)}%</div>`;
  return `<div class="score-badge">
    <div class="muted" style="font-size:11px">${esc(label)}</div>
    <div class="num" style="color:${color}">${score >= 0 ? '+' : ''}${score.toFixed(2)}</div>
    <div class="score-bar"><i style="width:${barPct}%;background:${color}"></i></div>
    ${confRow}
  </div>`;
}

function renderIndicatorTable(indicators){
  if (!indicators || !indicators.length){
    return '<div class="muted">Bu sembol için yeterli fiyat geçmişi olmadığından gösterge hesaplanamadı.</div>';
  }
  return `<table class="ind-table"><thead><tr><th>Gösterge</th><th>Değer</th><th>Yön</th><th>Ağırlık</th><th>Açıklama</th></tr></thead><tbody>
    ${indicators.map(ind => `<tr>
      <td><b>${esc(ind.name)}</b></td>
      <td>${esc(ind.value)}</td>
      <td class="dir${ind.direction}">${dirArrow(ind.direction)} ${dirLabel(ind.direction)}</td>
      <td>${ind.weight_pct}%</td>
      <td class="muted">${esc(ind.explanation)}</td>
      </tr>`).join('')}
  </tbody></table>`;
}

let detailChart = null;

async function openDetail(idx){
  const p = lastResults[idx];
  if (!p) return;
  document.getElementById('detailTitle').textContent = `${p.symbol} — ${dirLabel(p.final_direction)}`;
  document.getElementById('detailSub').textContent =
    `${esc(p.timeframe)} · ${esc(p.profile)} · İşlem fiyatı: ${formatPrice(p.symbol, p.price_at_prediction)}`;
  document.getElementById('detailScores').innerHTML =
    scoreBadge('Final Skor', p.final_score, p.final_confidence) +
    scoreBadge('Haber Skoru (AI)', p.news_score, p.news_confidence) +
    scoreBadge('Teknik Skor', p.technical_score, null);
  document.getElementById('detailNewsRationale').textContent = p.news_rationale || 'Yorum yok.';
  document.getElementById('detailIndicators').innerHTML = renderIndicatorTable(p.technical_indicators);
  document.getElementById('detailNews').innerHTML = '<div class="muted">Haberler yükleniyor...</div>';
  document.getElementById('detailModal').classList.add('open');

  const sources = p.news_sources || 'google';
  const newsUrl = `/api/news?symbol=${encodeURIComponent(p.symbol)}&name=${encodeURIComponent(p.symbol)}&sources=${encodeURIComponent(sources)}`;
  const chartUrl = `/api/chart?symbol=${encodeURIComponent(p.symbol)}&timeframe=${encodeURIComponent(p.timeframe || '1d')}`;

  const [newsResp, chartResp] = await Promise.all([fetch(newsUrl), fetch(chartUrl)]);
  const news = await newsResp.json();
  const chartData = await chartResp.json();

  document.getElementById('detailNews').innerHTML = news.length ? news.map(a => `
    <div class="news-item">
      <a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a>
      <div class="news-meta">${esc(a.source)} · ${new Date(a.published_ts).toLocaleString('tr-TR')}</div>
    </div>`).join('') : '<div class="muted">Bu sembol için haber bulunamadı.</div>';

  const labels = (chartData.dates || []).map(d => new Date(d).toLocaleDateString('tr-TR'));
  detailChart = chartOrUpdate(detailChart, document.getElementById('detailChartCanvas'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {label:'Kapanış', data:chartData.close, borderColor:'#6ca7ff', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:2},
        {label:'SMA50', data:chartData.sma50, borderColor:'#f4b942', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:1},
        {label:'SMA200', data:chartData.sma200, borderColor:'#ff6b6b', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:1},
        {label:'EMA20', data:chartData.ema20, borderColor:'#31c48d', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:1},
        {label:'Bollinger Üst', data:chartData.bb_upper, borderColor:'rgba(142,164,199,.5)', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:1, borderDash:[4,4]},
        {label:'Bollinger Alt', data:chartData.bb_lower, borderColor:'rgba(142,164,199,.5)', backgroundColor:'transparent', tension:.15, pointRadius:0, borderWidth:1, borderDash:[4,4]},
      ],
    },
    options: { responsive:true, plugins:{ legend:{labels:{color:'#e7eefb', boxWidth:12}} },
      scales:{ y:{ grid:{color:'rgba(255,255,255,.06)'} }, x:{ grid:{display:false}, ticks:{maxTicksLimit:8} } } },
  });
}

function closeDetail(){ document.getElementById('detailModal').classList.remove('open'); }

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') closeDetail();
});

async function loginUser(){
    const payload = {username: document.getElementById('auth_username').value, password: document.getElementById('auth_password').value};
    const r = await fetch('/api/login', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    if (!r.ok){
        const err = await r.json().catch(() => ({}));
        document.getElementById('auth_status').textContent = errDetail(err) || 'Giriş başarısız.';
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
        document.getElementById('auth_status').textContent = errDetail(err) || 'Kayıt başarısız.';
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
    document.getElementById('authCard').style.display = 'block';
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
        document.getElementById('authCard').style.display = 'block';
        document.getElementById('app_shell').style.display = 'none';
        document.getElementById('meStatus').textContent = 'oturum yok';
        return;
    }
    document.getElementById('meStatus').textContent = `giriş: ${me.username}`;
    document.getElementById('authCard').style.display = 'none';
    document.getElementById('app_shell').style.display = 'block';
    await loadSettings();
    await loadHistory();
    await loadDashboard();
}

initializeForSession();
</script></body></html>"""