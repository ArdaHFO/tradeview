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
from .news.fetch import available_sources, fetch_articles
from .pipeline import load_watchlist, run_for_symbols
from .screener import list_universes, scan as screener_scan, scan_symbols as screener_scan_symbols, universe_symbols
from .storage.recorder import PredictionRecorder
from .symbols_search import search_symbols
from .technical.data import ALLOWED_TIMEFRAMES, fetch_bars
from .technical.indicators import atr, bollinger_bands, ema, rsi, sma
from .technical.scorer import score_technical
from .learning.train import load_model

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


def _last_non_none(seq: list) -> float | None:
    for value in reversed(seq):
        if value is not None:
            return value
    return None


def _pivot_levels(high: float, low: float, close: float) -> dict:
    """Classic floor-trader pivots from the latest bar's H/L/C."""
    p = (high + low + close) / 3.0
    return {
        "p": round(p, 2),
        "r1": round(2 * p - low, 2), "s1": round(2 * p - high, 2),
        "r2": round(p + (high - low), 2), "s2": round(p - (high - low), 2),
    }


def _vwap_series(bars: list) -> list[float | None]:
    """Session-anchored VWAP: cumulative (typical price x volume) / cumulative
    volume, resetting at the start of every calendar day. Meaningful mainly on
    intraday timeframes (30m/1h) — traders use it as the session's "fair value"
    line. For daily+ bars each bar IS its own session, so it just reduces to
    that bar's typical price; the UI only offers the VWAP toggle intraday.
    """
    out: list[float | None] = []
    cum_pv = 0.0
    cum_vol = 0.0
    last_day = None
    for b in bars:
        day = b.ts.date()
        if day != last_day:
            cum_pv, cum_vol, last_day = 0.0, 0.0, day
        typical = (b.high + b.low + b.close) / 3.0
        cum_pv += typical * b.volume
        cum_vol += b.volume
        out.append(round(cum_pv / cum_vol, 4) if cum_vol > 0 else None)
    return out


def _fib_levels(high: float, low: float) -> dict:
    """Fibonacci retracement levels between the period high and low."""
    span = high - low
    return {label: round(high - span * pct, 2) for label, pct in (
        ("0", 0.0), ("23.6", 0.236), ("38.2", 0.382),
        ("50", 0.5), ("61.8", 0.618), ("100", 1.0),
    )}


def _chart_summary(closes: list[float], highs: list[float], lows: list[float],
                   rsi_series: list) -> dict:
    """Investor-friendly at-a-glance stats derived from the price series:
    last price, period change, range position, nearest support/resistance,
    RSI and ATR-based volatility. All cheap, all from data we already fetched.
    """
    if not closes:
        return {}
    last = closes[-1]
    prev = closes[-2] if len(closes) >= 2 else last
    period_high = max(closes)
    period_low = min(closes)
    span = period_high - period_low
    # Nearest support/resistance from recent swing lows/highs (last ~20 bars).
    window = min(20, len(closes))
    support = min(lows[-window:]) if lows else period_low
    resistance = max(highs[-window:]) if highs else period_high
    atr_series = atr(highs, lows, closes, 14) if len(closes) > 15 else []
    atr_last = _last_non_none(atr_series) if atr_series else None
    return {
        "last": last,
        "change_pct": ((last / prev) - 1.0) * 100.0 if prev else 0.0,
        "period_high": period_high,
        "period_low": period_low,
        "position_pct": ((last - period_low) / span * 100.0) if span > 0 else 50.0,
        "support": support,
        "resistance": resistance,
        "rsi": _last_non_none(rsi_series),
        "atr": atr_last,
        "atr_pct": (atr_last / last * 100.0) if atr_last and last else None,
        "pivot": _pivot_levels(highs[-1], lows[-1], closes[-1]) if highs and lows else None,
        "fib": _fib_levels(period_high, period_low) if span > 0 else None,
    }


def _prediction_to_dict(p: Prediction) -> dict:
    return {
        "ts": p.ts.isoformat(),
        "symbol": p.symbol,
        "name": p.name,
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

    @app.get("/api/news-sources")
    def api_news_sources(user_id: int = Depends(require_user)) -> JSONResponse:
        """Which news sources exist and whether each is usable right now
        (keyed sources need their API key configured)."""
        return JSONResponse(available_sources(_get_runtime(user_id).current_cfg()))

    @app.get("/api/screener/universes")
    def api_screener_universes(_: int = Depends(require_user)) -> JSONResponse:
        return JSONResponse(list_universes())

    @app.get("/api/stocks")
    def api_stocks(exchange: str = "bist", _: int = Depends(require_user)) -> JSONResponse:
        """Browsable list of the supported symbols for one exchange/universe."""
        return JSONResponse(universe_symbols(exchange))

    @app.get("/api/screener")
    def api_screener(universe: str = "bist", timeframe: str = "1d",
                     user_id: int = Depends(require_user)) -> JSONResponse:
        """Technical scan of a preset universe (or the user's watchlist), ranked
        by score, with the learned model's P(up) when a model is present. No
        Groq cost; runs synchronously."""
        current_cfg = _get_runtime(user_id).current_cfg()
        if universe == "watchlist":
            recorder = _recorder()
            try:
                rows = [dict(r) for r in recorder.list_watchlist(user_id=user_id)]
            finally:
                recorder.close()
            entries = [(r["symbol"], r.get("name") or r["symbol"]) for r in rows]
            return JSONResponse(screener_scan_symbols(entries, current_cfg, timeframe))
        return JSONResponse(screener_scan(universe, current_cfg, timeframe))

    @app.get("/api/news")
    def api_news(symbol: str, name: str = "", sources: str = "google",
                 user_id: int = Depends(require_user)) -> JSONResponse:
        """Raw article list for a symbol — no AI scoring, no Groq cost."""
        current_cfg = _get_runtime(user_id).current_cfg()
        source_list = [s.strip() for s in sources.split(",") if s.strip()]
        # This is the reader-facing article list, so ask fetch to widen the
        # window until it fills up (rather than stopping at the first recent hit).
        articles = fetch_articles(symbol, name or None, current_cfg, source_list,
                                  min_articles=current_cfg.max_articles_per_symbol)
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
        """Price series + indicator overlays + a compact stats summary for
        charting — no Groq cost."""
        if timeframe not in ALLOWED_TIMEFRAMES:
            timeframe = "1d"
        current_cfg = _get_runtime(user_id).current_cfg()
        bars = fetch_bars(symbol, current_cfg, timeframe)
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]
        sma50 = sma(closes, 50)
        sma200 = sma(closes, 200) if len(closes) >= 200 else [None] * len(closes)
        ema20 = ema(closes, 20)
        rsi14 = rsi(closes, 14)
        upper, mid, lower = bollinger_bands(closes)
        verdict = score_technical(symbol, bars)
        return JSONResponse({
            "symbol": symbol,
            "dates": [b.ts.isoformat() for b in bars],
            "open": [b.open for b in bars],
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "sma50": sma50,
            "sma200": sma200,
            "ema20": ema20,
            "rsi": rsi14,
            "bb_upper": upper,
            "bb_mid": mid,
            "bb_lower": lower,
            "vwap": _vwap_series(bars),
            "technical_score": round(verdict.score, 3),
            "technical_indicators": [
                {
                    "name": ind.name, "value": ind.value, "direction": ind.direction.value,
                    "weight_pct": ind.weight_pct, "explanation": ind.explanation,
                }
                for ind in verdict.indicators
            ],
            "summary": _chart_summary(closes, highs, lows, rsi14),
        })

    @app.get("/api/model")
    def api_model(user_id: int = Depends(require_user)) -> JSONResponse:
        """The trained learned-fusion model's honest, out-of-sample report card
        (or availability=false when no model is present)."""
        model = load_model(_get_runtime(user_id).current_cfg().model_path)
        if model is None:
            return JSONResponse({"available": False})
        return JSONResponse({
            "available": True,
            "meta": model.meta,
            "weights": dict(zip(model.feature_names, model.weights)),
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

    @app.delete("/api/watchlist/{symbol}")
    def api_watchlist_delete(symbol: str, user_id: int = Depends(require_user)) -> JSONResponse:
        recorder = _recorder()
        try:
            recorder.delete_watchlist(symbol, user_id=user_id)
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
<script src="https://cdn.jsdelivr.net/npm/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
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
.search-input-wrap{position:relative}
.dropdown{position:absolute;left:0;right:0;top:100%;margin-top:8px;background:var(--panel);border:1px solid var(--line);border-radius:14px;overflow:hidden;z-index:200;max-height:260px;overflow-y:auto;box-shadow:0 18px 40px rgba(0,0,0,.4)}
.dropdown .opt{padding:10px 14px;cursor:pointer;border-bottom:1px solid rgba(255,255,255,.04);display:flex;align-items:center;justify-content:space-between;gap:10px}
.dropdown .opt:hover,.dropdown .opt.active{background:rgba(108,167,255,.16)}
.dropdown .opt .sym{font-weight:700}
.dropdown .opt .nm{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;margin-left:4px}
.dropdown .ex{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:7px;padding:1px 8px;white-space:nowrap}
.dropdown .msg{padding:12px 14px;color:var(--muted);font-style:italic}
/* segmented toggle chips (timeframe / news sources) — replaces raw checkboxes */
.toggle-group{display:flex;flex-wrap:wrap;gap:8px}
.toggle{position:relative;display:inline-flex;align-items:center;gap:6px;padding:8px 13px;border-radius:999px;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted);cursor:pointer;user-select:none;font-size:13px;white-space:nowrap;transition:border-color .15s,background .15s,color .15s}
.toggle input{position:absolute;opacity:0;width:0;height:0}
.toggle:hover{border-color:#35558b;color:var(--text)}
.toggle:has(input:checked){background:rgba(108,167,255,.16);border-color:#4d7fd6;color:var(--text);font-weight:600}
.toggle:has(input:checked)::before{content:"✓";font-size:11px;color:var(--blue)}
.toggle:has(input:disabled){opacity:.4;cursor:not-allowed}
.toggle:has(input:disabled):hover{border-color:var(--line);color:var(--muted)}
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
/* native dropdown list was rendering white-on-white — force dark, readable options */
select,select option{color:var(--text);background:var(--panel2)}
select option{background:#13213a;color:#e7eefb}
table{width:100%;border-collapse:collapse}th,td{padding:11px 10px;border-bottom:1px solid rgba(255,255,255,.06);vertical-align:top;text-align:left}th{font-size:11px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}td{color:#dce7f8}
.up{color:var(--green);font-weight:700}.down{color:var(--red);font-weight:700}.neutral{color:var(--muted);font-weight:700}.reasons,.muted{color:var(--muted)}.empty{color:var(--muted);padding:18px 10px;font-style:italic;text-align:center}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend span{display:inline-flex;align-items:center;gap:6px}.dot{width:10px;height:10px;border-radius:999px;display:inline-block}
canvas{width:100%!important;height:340px!important}
/* TradingView Lightweight Charts containers */
.lwc{width:100%;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:rgba(255,255,255,.015)}
#lwPrice{height:460px}
@media (max-width:700px){#lwPrice{height:360px}}
.row-clickable{cursor:pointer}
.row-clickable:hover td{background:rgba(255,255,255,.03)}
.skel{background:linear-gradient(90deg, rgba(255,255,255,.04) 25%, rgba(255,255,255,.09) 37%, rgba(255,255,255,.04) 63%);background-size:400% 100%;animation:skel 1.4s ease infinite;border-radius:8px;height:14px}
@keyframes skel{0%{background-position:100% 50%}100%{background-position:0 50%}}
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
/* inline single-stock detail panel */
.detail-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.verdict{display:flex;align-items:center;gap:16px;flex-wrap:wrap;padding:14px 16px;border-radius:14px;border:1px solid var(--line);margin-bottom:14px;background:rgba(255,255,255,.03)}
.verdict.up{background:rgba(49,196,141,.12);border-color:rgba(49,196,141,.4)}
.verdict.down{background:rgba(255,107,107,.12);border-color:rgba(255,107,107,.4)}
.verdict.neutral{background:rgba(142,164,199,.1)}
.verdict .big{font-size:22px;font-weight:800;white-space:nowrap}
.verdict .txt{flex:1;min-width:220px;color:var(--text);line-height:1.5}
.verdict .conf{min-width:170px}
.conf-label{font-size:11px;color:var(--muted);margin-bottom:5px}
.conf-bar{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden}
.conf-bar i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--amber),var(--green))}
.statstrip{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:14px}
.statstrip .cell{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:12px;padding:10px 12px}
.statstrip .cell .k{font-size:11px;color:var(--muted)}
.statstrip .cell .v{font-size:16px;font-weight:700;margin-top:3px}
.pos-wrap{margin-bottom:8px}
.pos-track{position:relative;height:10px;border-radius:999px;background:linear-gradient(90deg,rgba(255,107,107,.4),rgba(244,185,66,.4),rgba(49,196,141,.4));margin:6px 0 5px}
.pos-track .marker{position:absolute;top:-4px;width:4px;height:18px;border-radius:3px;background:var(--text);transform:translateX(-50%);box-shadow:0 0 0 2px var(--panel)}
.pos-ends{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
.tech-summary{padding:11px 14px;border-radius:12px;background:rgba(255,255,255,.04);border:1px solid var(--line);margin-bottom:12px;line-height:1.5}
.ind-cards{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.ind-card{background:rgba(255,255,255,.03);border:1px solid var(--line);border-radius:12px;padding:12px 14px;border-left:3px solid var(--muted)}
.ind-card.up{border-left-color:var(--green)}.ind-card.down{border-left-color:var(--red)}.ind-card.neutral{border-left-color:var(--muted)}
.ind-card .top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.ind-card .nm{font-weight:700}
.ind-card .val{font-size:13px;font-weight:600}
.ind-card .exp{color:var(--muted);font-size:12px;margin-top:7px;line-height:1.5}
.ind-card .wbar{height:5px;border-radius:999px;background:rgba(255,255,255,.08);margin-top:9px;overflow:hidden}
.ind-card .wbar i{display:block;height:100%;background:var(--blue)}
tr.selected td{background:rgba(108,167,255,.14)!important}
.scr-select{background:rgba(255,255,255,.04);color:var(--text);border:1px solid var(--line);border-radius:12px;padding:9px 12px;font-size:13px;outline:none}
.sig{display:inline-flex;align-items:center;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;border:1px solid var(--line)}
.sig.buy2{color:var(--green);border-color:rgba(49,196,141,.5);background:rgba(49,196,141,.14)}
.sig.buy1{color:var(--green);border-color:rgba(49,196,141,.35)}
.sig.hold{color:var(--muted)}
.sig.sell1{color:var(--red);border-color:rgba(255,107,107,.35)}
.sig.sell2{color:var(--red);border-color:rgba(255,107,107,.5);background:rgba(255,107,107,.14)}
.levels{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin:4px 0 6px}
.levels .lv{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:10px;padding:8px 10px;text-align:center}
.levels .lv .k{font-size:10px;color:var(--muted)}
.levels .lv .v{font-size:13px;font-weight:700;margin-top:2px}
.chart-toggle{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}
.chart-toolbar{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.chart-toolbar .chart-toggle{margin-bottom:8px}
.toggle[title]{cursor:help}
.toggle:has(input:disabled){opacity:.4;cursor:not-allowed}
.chart-wrap{position:relative}
.chart-legend{position:absolute;top:10px;left:12px;right:12px;z-index:5;display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--text);background:rgba(8,16,30,.68);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);padding:7px 12px;border-radius:11px;pointer-events:none;border:1px solid rgba(255,255,255,.08)}
.chart-legend b{font-weight:700}
.chart-legend .muted{color:var(--muted)}
/* model weight bars (diverging from center) */
.wrow{display:flex;align-items:center;gap:10px;margin:7px 0}
.wrow .wl{width:96px;font-size:12px;color:var(--muted);text-align:right;flex:none}
.wrow .wt{flex:1;height:14px;background:rgba(255,255,255,.05);border-radius:6px;position:relative}
.wrow .wt .z{position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:rgba(255,255,255,.18)}
.wrow .wt i{position:absolute;top:0;height:100%;border-radius:5px}
.wrow .wv{width:52px;font-size:11px;font-weight:700;flex:none}
/* trade plan (buy/sell levels) */
.plan{width:100%;border-collapse:collapse;margin-top:4px}
.plan th,.plan td{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.06);text-align:left;font-size:13px}
.plan th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
.plan .tag-buy{color:var(--green);font-weight:800}
.plan .tag-sell{color:var(--red);font-weight:800}
.plan tr.cur td{background:rgba(108,167,255,.10);font-weight:700}
details.settings summary{cursor:pointer;font-size:15px;font-weight:700;list-style:none;display:flex;align-items:center;gap:8px}
details.settings summary::-webkit-details-marker{display:none}
details.settings summary::before{content:"▸";color:var(--muted);transition:transform .15s}
details.settings[open] summary::before{transform:rotate(90deg)}
/* tab navigation — sticky glass bar */
.tabs{position:sticky;top:10px;z-index:60;display:flex;gap:6px;flex-wrap:wrap;margin-bottom:18px;padding:6px;border:1px solid var(--line);border-radius:16px;background:rgba(10,20,38,.72);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 12px 30px rgba(0,0,0,.28)}
.tabs button{background:transparent;border:0;color:var(--muted);padding:10px 18px;font-size:14px;font-weight:600;cursor:pointer;border-radius:11px;transition:color .15s,background .15s}
.tabs button:hover{color:var(--text);background:rgba(255,255,255,.05)}
.tabs button.active{color:#eaf2ff;background:linear-gradient(180deg,rgba(108,167,255,.28),rgba(108,167,255,.14));box-shadow:inset 0 0 0 1px rgba(108,167,255,.35)}
.tabhide,.is-hidden{display:none!important}
/* modern polish */
h1{background:linear-gradient(92deg,#eaf2ff 20%,#8fb8ff 60%,#31c48d 110%);-webkit-background-clip:text;background-clip:text;color:transparent}
.grid > .card{animation:rise .35s ease both}
@keyframes rise{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.card{transition:border-color .2s,box-shadow .2s}
.card:hover{border-color:#2b4470}
table{font-variant-numeric:tabular-nums}
tbody tr{transition:background .12s}
tbody tr:hover td{background:rgba(255,255,255,.022)}
.btn{transition:transform .12s,filter .12s,box-shadow .12s}
.btn:hover:not(:disabled){transform:translateY(-1px);filter:brightness(1.07)}
.btn:active:not(:disabled){transform:translateY(0)}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#22365c;border-radius:999px}
::-webkit-scrollbar-track{background:transparent}
/* history filter chips + symbol performance bars */
.hist-chips{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.perfbar{display:flex;align-items:center;gap:8px}
.perfbar .track{flex:1;height:8px;border-radius:999px;background:rgba(255,255,255,.07);overflow:hidden}
.perfbar .track i{display:block;height:100%;border-radius:999px}
/* selective accuracy chips on the model card */
.sel-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}
.sel-cell{border:1px solid var(--line);border-radius:12px;padding:10px 12px;background:rgba(255,255,255,.03)}
.sel-cell.hot{border-color:rgba(49,196,141,.45);background:rgba(49,196,141,.08)}
.sel-cell .t{font-size:11px;color:var(--muted)}
.sel-cell .a{font-size:19px;font-weight:800;margin-top:2px}
.sel-cell .c{font-size:11px;color:var(--muted);margin-top:2px}
/* browse popular stocks */
.browse{margin-top:14px;border-top:1px solid rgba(255,255,255,.06);padding-top:12px}
.browse-list{display:flex;flex-wrap:wrap;gap:8px;max-height:190px;overflow-y:auto;margin-top:10px}
.stock-chip{display:inline-flex;flex-direction:column;gap:1px;padding:7px 12px;border-radius:12px;border:1px solid var(--line);background:rgba(255,255,255,.04);cursor:pointer;transition:border-color .15s,background .15s}
.stock-chip:hover{border-color:#4d7fd6;background:rgba(108,167,255,.14)}
.stock-chip .s{font-weight:700;font-size:13px}
.stock-chip .n{font-size:11px;color:var(--muted)}
/* result cards */
.result-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.rcard{background:linear-gradient(180deg,rgba(255,255,255,.045),rgba(255,255,255,.015));border:1px solid var(--line);border-radius:16px;padding:14px;cursor:pointer;transition:border-color .15s,transform .15s,box-shadow .15s;border-left:4px solid var(--muted)}
.rcard:hover{border-color:#35558b;transform:translateY(-2px);box-shadow:0 12px 28px rgba(0,0,0,.25)}
.rcard.up{border-left-color:var(--green)}.rcard.down{border-left-color:var(--red)}.rcard.neutral{border-left-color:var(--muted)}
.rcard.active{border-color:var(--blue);box-shadow:0 0 0 2px rgba(108,167,255,.3)}
.rcard .rc-head{display:flex;justify-content:space-between;align-items:center;gap:8px}
.rcard .rc-sym{font-size:17px;font-weight:800}
.rcard .rc-meta{font-size:11px;color:var(--muted);margin-top:2px}
.rcard .rc-scores{display:flex;gap:8px;margin-top:12px}
.rcard .rc-s{flex:1;background:rgba(255,255,255,.04);border-radius:10px;padding:7px 8px;text-align:center}
.rcard .rc-s .k{font-size:10px;color:var(--muted)}
.rcard .rc-s .v{font-size:15px;font-weight:700;margin-top:2px}
.rcard .rc-conf{margin-top:10px}
.rcard .rc-conf .cbar{height:6px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:4px}
.rcard .rc-conf .cbar i{display:block;height:100%;border-radius:999px;background:linear-gradient(90deg,var(--amber),var(--green))}
/* chart empty state + primary action bar */
.chart-empty{display:flex;align-items:center;justify-content:center;min-height:220px;color:var(--muted);font-style:italic;text-align:center;padding:0 24px}
.actionbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.btn.primary{background:linear-gradient(180deg,#7db0ff,#4d7fd6);color:#04101f;box-shadow:0 6px 16px rgba(108,167,255,.28)}
.action-hint{font-size:12px;color:var(--muted);margin-top:8px}
@media (max-width:1000px){.stat,.half{grid-column:span 12}.settings-grid{grid-template-columns:repeat(2,1fr)}.statstrip{grid-template-columns:repeat(3,1fr)}.ind-cards{grid-template-columns:1fr}.levels{grid-template-columns:repeat(3,1fr)}.result-cards{grid-template-columns:1fr}}
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

  <nav class="tabs" id="tabnav">
    <button data-goto="analiz" class="active" onclick="showTab('analiz')">🔍 Analiz</button>
    <button data-goto="tarayici" onclick="showTab('tarayici')">🔎 Tarayıcı</button>
    <button data-goto="panom" onclick="showTab('panom')">📊 Panom</button>
    <button data-goto="ayarlar" onclick="showTab('ayarlar')">⚙️ Ayarlar</button>
  </nav>

  <div class="grid">
    <div class="card stat" data-tab="panom"><div class="l">Son 30 Gün İsabet Oranı</div><div id="statHit" class="v">-</div></div>
    <div class="card stat" data-tab="panom"><div class="l">Favori Sayısı</div><div id="statWatchlist" class="v">-</div></div>
    <div class="card stat" data-tab="panom"><div class="l">Son Kullanılan Profil</div><div id="statModel" class="v">-</div></div>
    <div class="card stat" data-tab="panom"><div class="l">Son Kullanılan Zaman Dilimi</div><div id="statTf" class="v">-</div></div>

    <div class="card search" data-tab="analiz">
      <div class="row" style="justify-content:space-between;margin-bottom:10px">
        <div class="muted">Dünyanın herhangi bir borsasından sembol veya şirket adı ara (ör. AAPL, ASELS, THYAO, SAP, MC), seç ve analiz et.</div>
        <div class="legend"><span><i class="dot" style="background:var(--green)"></i> Yükseliş</span><span><i class="dot" style="background:var(--red)"></i> Düşüş</span><span><i class="dot" style="background:var(--muted)"></i> Nötr</span></div>
      </div>
      <div class="search-input-wrap">
        <input type="text" id="q" placeholder="AAPL, ASELS, THYAO, SAP, MC, Apple, Tesla..." autocomplete="off">
        <div id="dd" class="dropdown" style="display:none"></div>
      </div>
      <div class="browse">
        <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div class="muted" style="font-size:13px">📋 Adını hatırlamıyor musun? Desteklenen borsalardan göz atıp seç:</div>
          <div class="toggle-group" id="browse_ex"></div>
        </div>
        <div class="browse-list" id="browse_list"><span class="muted" style="font-size:12px">Yükleniyor…</span></div>
      </div>
      <div class="row" style="margin-top:12px;gap:16px;flex-wrap:wrap">
        <div class="field" style="min-width:220px">
          <label>Zaman dilimi (yeni eklenecek semboller için)</label>
          <div class="toggle-group" id="tf_controls">
            <label class="toggle"><input type="checkbox" class="tf_cb" value="1d" checked> 1 gün</label>
            <label class="toggle"><input type="checkbox" class="tf_cb" value="1h"> 1 saat</label>
            <label class="toggle"><input type="checkbox" class="tf_cb" value="30m"> 30 dk</label>
            <label class="toggle"><input type="checkbox" class="tf_cb" value="1wk"> 1 hafta</label>
            <label class="toggle"><input type="checkbox" class="tf_cb" value="1mo"> 1 ay</label>
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
            <option value="learned">🤖 Öğrenen model (≈3 ay momentum)</option>
          </select>
        </div>
        <div class="field" style="min-width:240px">
          <label>Haber kaynağı</label>
          <div class="toggle-group" id="src_controls"><span class="muted" style="font-size:12px">Yükleniyor…</span></div>
        </div>
      </div>
      <div style="margin-top:14px">
        <div class="chips" id="chips" style="margin-bottom:12px"></div>
        <div class="actionbar">
          <button class="btn primary" id="go" onclick="analyze()">▶ Analiz Et</button>
          <button class="btn alt analysis-btn" id="btnCompare" onclick="compareModels()" title="Seçili hisseleri 3 profille (dengeli / haber ağırlıklı / teknik ağırlıklı) analiz edip yan yana karşılaştırır">⚖ Profilleri Karşılaştır</button>
          <button class="btn alt analysis-btn" id="btnMulti" onclick="multiTimeframe()" title="Seçili zaman dilimlerinin her biri için ayrı analiz üretir (ör. 1 gün + 1 hafta)">⏱ Çoklu Zaman Dilimi</button>
          <span style="flex:1"></span>
          <button class="btn alt" onclick="saveWatchlist()" title="Seçili hisseleri favori listene kaydeder">★ Favorilere Ekle</button>
        </div>
        <div class="action-hint" id="actionHint">Bir veya daha fazla hisse seç, sonra <b>Analiz Et</b>. Karşılaştırma ve çoklu zaman dilimi de aynı seçimi kullanır.</div>
      </div>
      <div id="progress" class="sub" style="margin-top:10px"></div>
    </div>

    <div class="card panel" data-tab="tarayici">
      <div class="row" style="justify-content:space-between;margin-bottom:10px;flex-wrap:wrap;gap:10px">
        <div>
          <h3 style="margin:0">🔎 Piyasa Tarayıcı — Bugünün Sinyalleri</h3>
          <div class="sub">Bir piyasa evrenini teknik göstergelere göre tarar, en güçlü al/sat setup'larını sıralar. Haber/AI maliyeti yok.</div>
        </div>
        <div class="row" style="gap:10px;flex-wrap:wrap">
          <select id="scr_universe" class="scr-select"></select>
          <select id="scr_timeframe" class="scr-select">
            <option value="1d">1 gün</option>
            <option value="1wk">1 hafta</option>
            <option value="1h">1 saat</option>
          </select>
          <button class="btn good" id="scr_go" onclick="runScreener()">Tara</button>
        </div>
      </div>
      <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px">
        <div class="toggle-group" id="scr_filter">
          <label class="toggle"><input type="radio" name="scrf" value="all" checked> Tümü</label>
          <label class="toggle"><input type="radio" name="scrf" value="up"> 🟢 Yükseliş</label>
          <label class="toggle"><input type="radio" name="scrf" value="down"> 🔴 Düşüş</label>
        </div>
        <div class="row" style="gap:8px;align-items:center">
          <span class="muted" style="font-size:12px">Sırala:</span>
          <select id="scr_sort" class="scr-select" onchange="renderScreener()">
            <option value="score">Teknik skor</option>
            <option value="model">🤖 Model P↑</option>
            <option value="rsi">RSI</option>
          </select>
        </div>
      </div>
      <table><thead><tr><th>Sembol</th><th>Ad</th><th>Sinyal</th><th>Teknik</th><th>🤖 Model P↑</th><th>Fiyat</th><th>RSI</th><th></th></tr></thead>
      <tbody id="scr_results"><tr><td colspan="8" class="empty">"Tara" butonuna basarak seçili evreni tarayın.</td></tr></tbody></table>
      <div id="scr_status" class="sub" style="margin-top:8px"></div>
    </div>

        <details class="card panel settings" data-tab="ayarlar" open>
            <summary>⚙️ Uygulama Ayarları <span class="muted" style="font-weight:400;font-size:12px">— AI modeli, ağırlıklar, veri ufku (aç/kapat)</span></summary>
            <div class="settings-grid" style="margin-top:12px">
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
        </details>

    <div class="card panel" data-tab="panom">
      <h3 style="margin:0 0 10px">📈 Performans — Kümülatif İsabet Oranı</h3>
      <div id="hitEmpty" class="chart-empty is-hidden">Henüz sonuçlanmış tahmin yok. Analiz yaptıkça ve tahminler ertesi kapanışla eşleştikçe isabet eğrin burada oluşur.</div>
      <canvas id="hitChart"></canvas>
    </div>
    <div class="card panel" data-tab="panom">
      <h3 style="margin:0 0 10px">📊 Profil / Zaman Dilimi Bazında İsabet</h3>
      <div id="barEmpty" class="chart-empty is-hidden">Profil ve zaman dilimi bazında isabet oranları, yeterli sonuçlanmış tahmin biriktiğinde görünür.</div>
      <canvas id="barChart"></canvas>
    </div>

    <div class="card panel" data-tab="panom" id="modelCard">
      <div class="row" style="justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px">
        <h3 style="margin:0">🤖 Yapay Zekâ Modeli — Öğrenme Sonucu</h3>
        <span class="muted" style="font-size:12px">≈3 aylık yön modeli · gerçek (out-of-sample) sonuçlar</span>
      </div>
      <div id="modelBody"><div class="muted">Yükleniyor…</div></div>
    </div>

    <div class="card panel is-hidden" id="compareCard" data-tab="analiz">
      <h3 style="margin:0 0 4px">⚖ Hisse Karşılaştırma</h3>
      <div class="sub" style="margin-bottom:10px">Bu çalıştırmada analiz edilen semboller yan yana.</div>
      <canvas id="compareChartCanvas" style="margin-bottom:14px"></canvas>
      <table><thead><tr><th>Sembol</th><th>Yön</th><th>Final Skor</th><th>Haber Skoru</th><th>Teknik Skor</th><th title="Yön tahmininin doğru olma ihtimali değil — sinyallerin ne kadar güçlü/net konuştuğunu gösterir.">Sinyal Gücü ℹ️</th></tr></thead>
      <tbody id="compareTable"></tbody></table>
    </div>

    <div class="card panel" data-tab="analiz"><h3 style="margin:0 0 10px">🔍 Son Analiz Sonuçları <span class="muted" style="font-weight:400;font-size:12px">— bir karta tıklayınca aşağıda grafikleri ve göstergeleri açar</span></h3>
      <div id="results" class="result-cards"><div class="empty" style="grid-column:1/-1">Henüz analiz yok — yukarıdan bir sembol seçip "Analiz Et" butonuna basın.</div></div>
    </div>

    <div class="card panel is-hidden" id="detailPanel" data-tab="analiz">
      <div class="detail-head">
        <div>
          <h3 id="detailTitle" style="margin:0">—</h3>
          <div id="detailSub" class="sub"></div>
        </div>
        <div id="detailSwitch" class="muted" style="font-size:12px"></div>
      </div>
      <div id="detailVerdict"></div>
      <div id="detailSummary" class="statstrip"></div>
      <div class="score-badges" id="detailScores"></div>
      <div id="detailPosition" class="pos-wrap"></div>

      <div class="section-title">📈 Fiyat Grafiği <span class="muted" style="font-weight:400;font-size:12px">— göstergeleri istediğin gibi aç/kapat</span></div>
      <div class="chart-toolbar">
        <div class="chart-toggle">
          <label class="toggle"><input type="radio" name="charttype" value="line" checked onchange="onChartPrefChange()"> Çizgi</label>
          <label class="toggle"><input type="radio" name="charttype" value="candle" onchange="onChartPrefChange()"> Mum</label>
        </div>
        <div class="chart-toggle">
          <label class="toggle"><input type="radio" name="pricescale" value="linear" checked onchange="onChartPrefChange()"> Doğrusal</label>
          <label class="toggle" title="Uzun vadeli/yüksek volatiliteli grafiklerde yüzdesel değişimi doğru gösterir"><input type="radio" name="pricescale" value="log" onchange="onChartPrefChange()"> Logaritmik</label>
        </div>
        <div class="chart-toggle" id="chartTfSwitch">
          <label class="toggle"><input type="radio" name="charttf" value="30m" onchange="switchChartTimeframe('30m')"> 30dk</label>
          <label class="toggle"><input type="radio" name="charttf" value="1h" onchange="switchChartTimeframe('1h')"> 1s</label>
          <label class="toggle"><input type="radio" name="charttf" value="1d" checked onchange="switchChartTimeframe('1d')"> 1G</label>
          <label class="toggle"><input type="radio" name="charttf" value="1wk" onchange="switchChartTimeframe('1wk')"> 1H</label>
          <label class="toggle"><input type="radio" name="charttf" value="1mo" onchange="switchChartTimeframe('1mo')"> 1A</label>
        </div>
      </div>
      <div class="chart-toggle" id="indicatorToggles">
        <label class="toggle"><input type="checkbox" data-ind="sma50" checked onchange="onChartPrefChange()"> SMA 50</label>
        <label class="toggle"><input type="checkbox" data-ind="sma200" checked onchange="onChartPrefChange()"> SMA 200</label>
        <label class="toggle"><input type="checkbox" data-ind="ema20" checked onchange="onChartPrefChange()"> EMA 20</label>
        <label class="toggle"><input type="checkbox" data-ind="bb" checked onchange="onChartPrefChange()"> Bollinger</label>
        <label class="toggle" title="Sadece gün içi (30dk/1s) zaman diliminde anlamlıdır"><input type="checkbox" data-ind="vwap" onchange="onChartPrefChange()"> VWAP</label>
        <label class="toggle"><input type="checkbox" data-ind="volume" checked onchange="onChartPrefChange()"> Hacim</label>
        <label class="toggle"><input type="checkbox" data-ind="sr" checked onchange="onChartPrefChange()"> Destek/Direnç</label>
        <label class="toggle"><input type="checkbox" data-ind="pivot" onchange="onChartPrefChange()"> Pivot</label>
        <label class="toggle"><input type="checkbox" data-ind="fib" onchange="onChartPrefChange()"> Fibonacci</label>
      </div>
      <div class="chart-wrap">
        <div id="lwPrice" class="lwc"></div>
        <div id="chartLegend" class="chart-legend"></div>
      </div>

      <div class="section-title" style="margin-top:14px">📐 Önemli Seviyeler <span class="muted" style="font-weight:400;font-size:12px">— pivot (son bar) ve Fibonacci geri çekilme</span></div>
      <div id="detailLevels"></div>

      <div class="section-title" style="margin-top:14px">🎯 İşlem Planı — Nereden Al, Nereden Sat <span class="muted" style="font-weight:400;font-size:12px">— destek/direnç · pivot · Fibonacci seviyelerinden üretildi</span></div>
      <div id="detailPlan"></div>

      <div class="section-title">🧮 Teknik Göstergeler</div>
      <div id="detailTechSummary"></div>
      <div id="detailIndicators"></div>

      <div class="section-title">💬 Yapay Zekâ Haber Yorumu</div>
      <div id="detailNewsRationale" class="muted"></div>

      <div class="section-title">📰 İlgili Haberler</div>
      <div id="detailNews"><div class="muted">Yükleniyor...</div></div>
    </div>

    <div class="card half" data-tab="panom"><h3 style="margin:0 0 10px">★ Favori Listem</h3><table><thead><tr><th>Sembol</th><th>Ad</th><th>Profil / Zaman Dilimi</th><th style="text-align:right">İşlem</th></tr></thead><tbody id="watchlist"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table></div>
    <div class="card half" data-tab="panom">
      <h3 style="margin:0 0 10px">🕓 Geçmiş Tahminler <span class="muted" style="font-weight:400;font-size:12px">— satıra tıkla, detayı aç</span></h3>
      <div class="hist-chips toggle-group" id="hist_filter">
        <label class="toggle"><input type="radio" name="histf" value="all" checked> Tümü</label>
        <label class="toggle"><input type="radio" name="histf" value="hit"> ✅ İsabet</label>
        <label class="toggle"><input type="radio" name="histf" value="miss"> ❌ Kaçırdı</label>
        <label class="toggle"><input type="radio" name="histf" value="pending"> ⏳ Bekliyor</label>
      </div>
      <table><thead><tr><th>Zaman</th><th>Sembol</th><th>Yön</th><th>İsabet</th></tr></thead><tbody id="history"><tr><td colspan="4" class="empty">Yükleniyor...</td></tr></tbody></table>
    </div>
    <div class="card panel" data-tab="panom">
      <h3 style="margin:0 0 10px">🏆 Sembol Performansı <span class="muted" style="font-weight:400;font-size:12px">— son 30 gün, sonuçlanmış tahminlere göre</span></h3>
      <table><thead><tr><th>Sembol</th><th>Tahmin</th><th>İsabet</th><th style="width:45%">Oran</th><th title="Yön tahmininin doğru olma ihtimali değil — sinyallerin ortalama gücü.">Ort. Sinyal Gücü ℹ️</th></tr></thead><tbody id="symperf"><tr><td colspan="5" class="empty">Henüz sonuçlanmış tahmin yok.</td></tr></tbody></table>
    </div>
    </div>

    </div>
</div>

<script>
let selected = [];
let pollTimer = null;
let hitChart = null;
let barChart = null;
let appSettings = {};

// Make Chart.js legible on the dark theme (bigger, brighter, point-style legends).
if (window.Chart){
  Chart.defaults.color = '#c9d6ea';
  Chart.defaults.font.size = 13;
  Chart.defaults.font.family = 'Inter, Segoe UI, sans-serif';
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.tooltip.titleFont = {size: 13};
  Chart.defaults.plugins.tooltip.bodyFont = {size: 13};
  Chart.defaults.plugins.tooltip.padding = 10;
}

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
const ddEl = document.getElementById('dd');
let searchDebounce = null;
let lastSearchResults = [];
let activeIdx = -1;
let searchSeq = 0;          // guards against out-of-order responses

function exchangeBadge(sym){
  return (sym || '').includes('.') ? sym.split('.').pop().toUpperCase() : 'US';
}
function hideDropdown(){ ddEl.style.display = 'none'; activeIdx = -1; }
function showMessage(html){ ddEl.innerHTML = `<div class="msg">${html}</div>`; ddEl.style.display = 'block'; activeIdx = -1; }

function renderDropdown(){
  if (!lastSearchResults.length){ showMessage('Sonuç bulunamadı — farklı bir kod ya da şirket adı deneyin.'); return; }
  ddEl.innerHTML = lastSearchResults.map((it, i) => `
    <div class="opt${i === activeIdx ? ' active' : ''}" data-idx="${i}">
      <span class="sym">${esc(it.symbol)}</span>
      <span class="nm">${esc(it.name)}</span>
      <span class="ex">${esc(exchangeBadge(it.symbol))}</span>
    </div>`).join('');
  ddEl.style.display = 'block';
}

async function runSearch(){
  const q = qEl.value.trim();
  if (q.length < 2) { hideDropdown(); lastSearchResults = []; return; }
  const seq = ++searchSeq;
  showMessage('Aranıyor…');
  let items = [];
  try {
    const r = await fetch('/api/symbols?q=' + encodeURIComponent(q));
    items = await r.json();
  } catch (e) { items = []; }
  if (seq !== searchSeq) return;   // a newer keystroke already fired
  lastSearchResults = Array.isArray(items) ? items : [];
  activeIdx = lastSearchResults.length ? 0 : -1;
  renderDropdown();
}

qEl.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(runSearch, 250);
});

qEl.addEventListener('keydown', (ev) => {
  const open = ddEl.style.display === 'block' && lastSearchResults.length;
  if (ev.key === 'ArrowDown' && open){
    ev.preventDefault();
    activeIdx = (activeIdx + 1) % lastSearchResults.length;
    renderDropdown();
  } else if (ev.key === 'ArrowUp' && open){
    ev.preventDefault();
    activeIdx = (activeIdx - 1 + lastSearchResults.length) % lastSearchResults.length;
    renderDropdown();
  } else if (ev.key === 'Enter' && open){
    ev.preventDefault();
    pick(lastSearchResults[activeIdx >= 0 ? activeIdx : 0]);
  } else if (ev.key === 'Escape'){
    hideDropdown();
  }
});

ddEl.addEventListener('click', (ev) => {
  const opt = ev.target.closest('[data-idx]');
  if (!opt) return;
  const item = lastSearchResults[Number(opt.dataset.idx)];
  if (item) pick(item);
});

document.addEventListener('click', (ev) => {
  if (ev.target !== qEl && !ddEl.contains(ev.target)) hideDropdown();
});

// Re-render the screener table when the up/down/all filter changes.
document.getElementById('scr_filter').addEventListener('change', renderScreener);
// Re-render history when its hit/miss/pending filter changes.
document.getElementById('hist_filter').addEventListener('change', renderHistory);

function checkedValues(selector){
  return [...document.querySelectorAll(selector + ':checked')].map(el => el.value);
}
function controlTimeframes(){ return checkedValues('.tf_cb').join(',') || '1d'; }
function controlSources(){ return checkedValues('.src_cb'); }
function controlProfile(){ return document.getElementById('profile_control').value || 'balanced'; }

const DEFAULT_SOURCES = ['google', 'yahoo'];
async function loadNewsSources(){
  const box = document.getElementById('src_controls');
  let sources = [];
  try {
    const r = await fetch('/api/news-sources');
    if (r.ok) sources = await r.json();
  } catch (e) { sources = []; }
  if (!sources.length){
    // Fall back to the always-keyless sources if the endpoint is unavailable.
    sources = [{id:'google', label:'Google Haberler', available:true, needs_key:false},
               {id:'yahoo', label:'Yahoo Finans', available:true, needs_key:false}];
  }
  box.innerHTML = sources.map(s => {
    const checked = s.available && DEFAULT_SOURCES.includes(s.id) ? ' checked' : '';
    const disabled = s.available ? '' : ' disabled';
    const hint = s.available ? '' : ' — API anahtarı gerekli';
    const title = s.needs_key && !s.available ? ' title="Sunucuda API anahtarı ayarlanınca kullanılabilir"' : '';
    return `<label class="toggle"${title}><input type="checkbox" class="src_cb" value="${esc(s.id)}"${checked}${disabled}> ${esc(s.label)}${hint}</label>`;
  }).join('');
}

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
  lastSearchResults = [];
  hideDropdown();
  qEl.focus();
  renderChips();
}

function remove(symbol){ selected = selected.filter(s => s.symbol !== symbol); renderChips(); }
function renderChips(){
  const c = document.getElementById('chips');
  if (!selected.length){
    c.innerHTML = '<span class="muted" style="font-size:12px">Henüz sembol seçilmedi — yukarıdaki kutudan arayıp ekleyin.</span>';
    return;
  }
  c.innerHTML = selected.map(s =>
    `<span class="chip" title="${esc(s.name || s.symbol)} · ${esc((s.news_sources||[]).join(', '))}">
      <span><b>${esc(s.symbol)}</b> <span class="muted" style="font-size:11px">${esc(s.timeframe)} · ${esc(s.profile)}</span></span>
      <button aria-label="kaldır" onclick="remove('${esc(s.symbol)}')">×</button></span>`).join('');
}

async function postAnalyze(url, payload){
  // Immediate feedback so it's obvious the click registered.
  document.getElementById('go').disabled = true;
  document.querySelectorAll('.analysis-btn').forEach(b => b.disabled = true);
  document.getElementById('progress').textContent = 'Başlatılıyor…';
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  if (!r.ok){
    const err = await r.json().catch(() => ({}));
    document.getElementById('go').disabled = false;
    document.querySelectorAll('.analysis-btn').forEach(b => b.disabled = false);
    document.getElementById('progress').textContent = '';
    alert(errDetail(err) || 'İşlem başlatılamadı.');
    return false;
  }
  const data = await r.json();
  if (data.started){ showTab('analiz'); startPolling(); }
  return true;
}

function selectedPayload(){
  return {symbols: selected.map(s => ({symbol:s.symbol, name:s.name, timeframe:s.timeframe, profile:s.profile, news_sources:s.news_sources}))};
}

function requireSelection(){
  if (selected.length) return true;
  document.getElementById('actionHint').innerHTML = '⚠️ Önce en az bir hisse seç (yukarıdaki kutudan ara ya da borsa listesinden tıkla).';
  return false;
}
async function analyze(){ if (requireSelection()) await postAnalyze('/api/analyze', selectedPayload()); }
async function compareModels(){ if (requireSelection()) await postAnalyze('/api/analyze/compare', selectedPayload()); }
async function multiTimeframe(){ if (requireSelection()) await postAnalyze('/api/analyze/multi', selectedPayload()); }

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
    showTab('analiz');
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

// ---- Tab navigation ----
function showTab(name){
  document.querySelectorAll('#tabnav button').forEach(b => b.classList.toggle('active', b.dataset.goto === name));
  document.querySelectorAll('[data-tab]').forEach(el => el.classList.toggle('tabhide', el.dataset.tab !== name));
  // A chart drawn while its tab was hidden has zero size — resize on reveal.
  requestAnimationFrame(() => {
    [hitChart, barChart, compareChart].forEach(c => { if (c) c.resize(); });
    resizeLwCharts();
  });
}

// ---- Browse supported exchanges (for when a ticker isn't recalled) ----
let browseExchange = 'bist';
async function loadBrowseExchanges(){
  const box = document.getElementById('browse_ex');
  let list = [];
  try { const r = await fetch('/api/screener/universes'); if (r.ok) list = await r.json(); } catch (e) {}
  if (!list.length) list = [{id:'bist', label:'BIST'}, {id:'us', label:'ABD'}, {id:'eu', label:'Avrupa'}];
  box.innerHTML = list.map((u, i) => `<label class="toggle"><input type="radio" name="brx" value="${esc(u.id)}" ${i === 0 ? 'checked' : ''} onchange="selectBrowseExchange('${esc(u.id)}')"> ${esc(u.label.replace(' Popüler', ''))}</label>`).join('');
  browseExchange = list[0].id;
  loadBrowseStocks(browseExchange);
}
function selectBrowseExchange(ex){ browseExchange = ex; loadBrowseStocks(ex); }
async function loadBrowseStocks(ex){
  const box = document.getElementById('browse_list');
  box.innerHTML = '<span class="muted" style="font-size:12px">Yükleniyor…</span>';
  let list = [];
  try { const r = await fetch('/api/stocks?exchange=' + encodeURIComponent(ex)); if (r.ok) list = await r.json(); } catch (e) {}
  if (!list.length){ box.innerHTML = '<span class="muted" style="font-size:12px">Liste alınamadı.</span>'; return; }
  box.innerHTML = list.map(s => `<div class="stock-chip" onclick='pick(${JSON.stringify({symbol:s.symbol, name:s.name}).replace(/'/g, "&#39;")})'>
    <span class="s">${esc(s.symbol)}</span><span class="n">${esc(s.name)}</span></div>`).join('');
}

// ---- Market screener ----
let screenerResults = [];

async function loadUniverses(){
  const sel = document.getElementById('scr_universe');
  try {
    const r = await fetch('/api/screener/universes');
    const list = await r.json();
    sel.innerHTML = list.map(u => `<option value="${esc(u.id)}">${esc(u.label)} · ${u.count}</option>`).join('')
      + '<option value="watchlist">★ Favorilerim</option>';
  } catch (e) { sel.innerHTML = '<option value="bist">BIST Popüler</option><option value="watchlist">★ Favorilerim</option>'; }
}

function sigClass(signal){
  return {'Güçlü Al':'buy2', 'Al':'buy1', 'Nötr':'hold', 'Sat':'sell1', 'Güçlü Sat':'sell2'}[signal] || 'hold';
}
function screenerFilterValue(){ const el = document.querySelector('input[name="scrf"]:checked'); return el ? el.value : 'all'; }

async function runScreener(){
  const btn = document.getElementById('scr_go');
  const universe = document.getElementById('scr_universe').value;
  const tf = document.getElementById('scr_timeframe').value;
  btn.disabled = true;
  document.getElementById('scr_status').textContent = 'Taranıyor… (birkaç saniye sürebilir)';
  document.getElementById('scr_results').innerHTML = skeletonRows(8, 6);
  try {
    const r = await fetch(`/api/screener?universe=${encodeURIComponent(universe)}&timeframe=${encodeURIComponent(tf)}`);
    screenerResults = await r.json();
  } catch (e) { screenerResults = []; }
  btn.disabled = false;
  const hasModel = screenerResults.some(r => r.model_proba != null);
  document.getElementById('scr_status').textContent =
    screenerResults.length ? `${screenerResults.length} sembol tarandı.${hasModel ? ' 🤖 Model P↑ = ~3 aylık yükseliş olasılığı.' : ''}` : 'Sonuç alınamadı (bu evren/favori listesi boş olabilir).';
  renderScreener();
}

function screenerSortValue(){ const el = document.getElementById('scr_sort'); return el ? el.value : 'score'; }

function renderScreener(){
  const f = screenerFilterValue();
  const sortKey = screenerSortValue();
  let rows = screenerResults.filter(r => f === 'all'
    || (f === 'up' && r.direction === 'UP') || (f === 'down' && r.direction === 'DOWN'));
  const keyOf = (r) => sortKey === 'model' ? (r.model_proba == null ? -1 : r.model_proba)
    : sortKey === 'rsi' ? (r.rsi == null ? -1 : r.rsi) : r.score;
  rows = rows.slice().sort((a, b) => keyOf(b) - keyOf(a));
  const tb = document.getElementById('scr_results');
  if (!rows.length){ tb.innerHTML = '<tr><td colspan="8" class="empty">Bu filtrede sonuç yok.</td></tr>'; return; }
  const modelCell = (r) => {
    if (r.model_proba == null) return '<span class="muted">—</span>';
    const col = r.model_proba >= 55 ? 'var(--green)' : (r.model_proba <= 45 ? 'var(--red)' : 'var(--muted)');
    return `<b style="color:${col}">%${r.model_proba.toFixed(0)}</b>`;
  };
  tb.innerHTML = rows.map(r => `<tr>
    <td><b>${esc(r.symbol)}</b></td>
    <td class="muted">${esc(r.name || '')}</td>
    <td><span class="sig ${sigClass(r.signal)}">${esc(r.signal)}</span></td>
    <td class="${dirClass(r.direction)}">${r.score >= 0 ? '+' : ''}${r.score.toFixed(2)}</td>
    <td>${modelCell(r)}</td>
    <td>${formatPrice(r.symbol, r.price)}</td>
    <td>${r.rsi == null ? '—' : r.rsi.toFixed(0)}</td>
    <td><button class="btn alt small" onclick='analyzeFromScreener(${JSON.stringify({symbol:r.symbol, name:r.name}).replace(/'/g, "&#39;")})'>＋ Analiz</button></td>
  </tr>`).join('');
}

function analyzeFromScreener(item){
  if (!selected.some(s => s.symbol === item.symbol)){
    selected.push({
      symbol: item.symbol, name: item.name,
      timeframe: controlTimeframes(), profile: controlProfile(),
      news_sources: controlSources().length ? controlSources() : ['google'],
    });
    renderChips();
  }
  analyze();
}

function startPolling(){
  // A fresh run: forget the previously spotlighted result and hide the panel
  // until new results arrive.
  currentDetailIdx = -1;
  const panel = document.getElementById('detailPanel');
  if (panel) panel.classList.add('is-hidden');
  if (pollTimer) return;
  pollTimer = setInterval(refreshState, 1500);
  refreshState();
}

let lastResults = [];

function skeletonRows(cols, rows){
  const cells = Array.from({length: cols}, () => `<td><div class="skel"></div></td>`).join('');
  return Array.from({length: rows}, () => `<tr>${cells}</tr>`).join('');
}

function skeletonCards(n){
  return Array.from({length: n}, () => `<div class="rcard"><div class="skel" style="height:20px;width:55%"></div><div class="skel" style="height:44px;margin-top:14px"></div><div class="skel" style="height:12px;margin-top:12px;width:70%"></div></div>`).join('');
}

async function refreshState(){
  const r = await fetch('/api/state');
  const s = await r.json();
  const st = document.getElementById('status');
  st.textContent = statusLabel(s.status) + (s.progress ? ' · ' + s.progress : '');
  st.className = 'pill ' + (s.status === 'error' ? 'danger' : (s.status === 'running' ? 'warn' : 'good'));
  const running = s.status === 'running';
  document.getElementById('go').disabled = running;
  document.querySelectorAll('.analysis-btn').forEach(b => b.disabled = running);
  document.getElementById('progress').textContent = s.error ? ('Hata: ' + s.error) : (s.progress || '');

  const results = document.getElementById('results');
  if (running && !s.results.length){
    results.innerHTML = skeletonCards(3);
  } else if (s.results.length){
    lastResults = s.results;
    results.innerHTML = s.results.map((p, i) => `<div id="resrow-${i}" class="rcard ${dirClass(p.final_direction)}" onclick="openDetail(${i}, true)">
      <div class="rc-head">
        <div><div class="rc-sym">${esc(p.symbol)}</div><div class="rc-meta">${esc(p.timeframe)} · ${esc(p.profile)}</div></div>
        <span class="pill ${p.final_direction === 'UP' ? 'good' : (p.final_direction === 'DOWN' ? 'danger' : '')}">${dirArrow(p.final_direction)} ${dirLabel(p.final_direction)}</span>
      </div>
      <div class="rc-scores">
        <div class="rc-s"><div class="k">Final</div><div class="v" style="color:${scoreColor(p.final_score)}">${p.final_score >= 0 ? '+' : ''}${p.final_score.toFixed(2)}</div></div>
        <div class="rc-s"><div class="k">Haber</div><div class="v" style="color:${scoreColor(p.news_score)}">${p.news_score >= 0 ? '+' : ''}${p.news_score.toFixed(2)}</div></div>
        <div class="rc-s"><div class="k">Teknik</div><div class="v" style="color:${scoreColor(p.technical_score)}">${p.technical_score >= 0 ? '+' : ''}${p.technical_score.toFixed(2)}</div></div>
      </div>
      <div class="rc-conf"><div class="muted" style="font-size:11px" title="Yön tahmininin doğru olma ihtimali değil — sinyal gücü.">${p.profile === 'learned' ? 'Model Güveni' : 'Sinyal Gücü'} %${Math.round(p.final_confidence * 100)} ℹ️</div><div class="cbar"><i style="width:${Math.round(p.final_confidence * 100)}%"></i></div></div>
    </div>`).join('');
    renderComparison(s.results);
    // Auto-open the first result so charts + indicators show without a click.
    if (currentDetailIdx < 0 || currentDetailIdx >= s.results.length){
      openDetail(0, false);
    } else {
      highlightResultRow(currentDetailIdx);
    }
  } else if (s.status === 'done'){
    results.innerHTML = '<div class="empty" style="grid-column:1/-1">Sonuç bulunamadı — sembollerin fiyat verisi alınamamış olabilir.</div>';
  }
  if (s.status !== 'running' && pollTimer){ clearInterval(pollTimer); pollTimer = null; loadHistory(); loadDashboard(); }
}

async function loadHistory(){
  // statHit ("Son 30 gün hit rate") is exclusively loadDashboard's job —
  // this used to also set it from a 7-day window under the same label,
  // and whichever of the two async calls finished last silently won.
  const r = await fetch('/api/history?days=7');
  const s = await r.json();
  historyRows = s.recent || [];
  renderHistory();
}

let historyRows = [];
function historyFilterValue(){ const el = document.querySelector('input[name="histf"]:checked'); return el ? el.value : 'all'; }
function renderHistory(){
  const f = historyFilterValue();
  const rows = historyRows.filter(p => f === 'all'
    || (f === 'hit' && p.hit === 1) || (f === 'miss' && p.hit === 0) || (f === 'pending' && p.hit === null));
  document.getElementById('history').innerHTML = rows.length ? rows.map(p => `<tr class="row-clickable" onclick='openHistoryDetail(${JSON.stringify(p).replace(/'/g, "&#39;")})'>
    <td>${new Date(p.ts).toLocaleString('tr-TR')}</td>
    <td><b>${esc(p.symbol)}</b></td>
    <td class="${dirClass(p.final_direction)}">${dirLabel(p.final_direction)}</td>
    <td>${p.hit === null ? '⏳' : (p.hit ? '✅' : '❌')}</td>
    </tr>`).join('') : '<tr><td colspan="4" class="empty">Bu filtrede tahmin yok.</td></tr>';
}

function chartOrUpdate(current, ctx, config){ if (current) current.destroy(); return new Chart(ctx, config); }
function toggleChartEmpty(canvasId, emptyId, hasData){
  document.getElementById(canvasId).classList.toggle('is-hidden', !hasData);
  document.getElementById(emptyId).classList.toggle('is-hidden', hasData);
}

async function loadDashboard(){
  const r = await fetch('/api/dashboard?days=30');
  const d = await r.json();
  document.getElementById('statHit').textContent = d.hit_rate.total > 0 ? `${d.hit_rate.hits}/${d.hit_rate.total} (${pct(d.hit_rate.hits, d.hit_rate.total)})` : 'Veri yok';
  document.getElementById('statWatchlist').textContent = d.watchlist.length;
  document.getElementById('statModel').textContent = d.by_profile[0] ? d.by_profile[0].profile : '-';
  document.getElementById('statTf').textContent = d.by_timeframe[0] ? d.by_timeframe[0].timeframe : '-';

  // Cumulative hit-rate curve — with a soft gradient fill and a friendly empty
  // state so a fresh account (no resolved predictions yet) doesn't show a blank.
  const hitHasData = (d.hit_series || []).length > 0;
  toggleChartEmpty('hitChart', 'hitEmpty', hitHasData);
  if (hitHasData){
    const hcx = document.getElementById('hitChart').getContext('2d');
    const grad = hcx.createLinearGradient(0, 0, 0, 300);
    grad.addColorStop(0, 'rgba(108,167,255,.38)');
    grad.addColorStop(1, 'rgba(108,167,255,0)');
    hitChart = chartOrUpdate(hitChart, document.getElementById('hitChart'), {
      type:'line',
      data:{ labels:d.hit_series.map(x => new Date(x.ts).toLocaleDateString('tr-TR')), datasets:[{label:'Kümülatif İsabet Oranı (%)', data:d.hit_series.map(x => x.running_hit_rate), borderColor:'#6ca7ff', backgroundColor:grad, borderWidth:2.5, tension:.35, fill:true, pointRadius:0, pointHoverRadius:5, pointHoverBackgroundColor:'#6ca7ff' }]},
      options:{ responsive:true, interaction:{mode:'index', intersect:false}, plugins:{ legend:{display:false}, tooltip:{callbacks:{label:(c)=>'İsabet: %'+c.parsed.y}} }, scales:{ y:{ beginAtZero:true, max:100, ticks:{callback:(v)=>v+'%'}, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
    });
  } else if (hitChart){ hitChart.destroy(); hitChart = null; }

  // Hit *rate* (%), not raw hit *count* — a profile run 40 times isn't
  // "better" than one run 5 times just because it has more hits.
  const rate = (x) => x.total > 0 ? Math.round(100 * x.hits / x.total) : 0;
  const profileMap = new Map((d.by_profile || []).map(x => [x.profile, rate(x)]));
  const timeframeMap = new Map((d.by_timeframe || []).map(x => [x.timeframe, rate(x)]));
  const labels = [...new Set([...profileMap.keys(), ...timeframeMap.keys()])].filter(Boolean);
  toggleChartEmpty('barChart', 'barEmpty', labels.length > 0);
  if (labels.length){
    barChart = chartOrUpdate(barChart, document.getElementById('barChart'), {
      type:'bar',
      data:{
        labels,
        datasets:[
          {label:'Profil isabet %', data:labels.map(l => profileMap.has(l) ? profileMap.get(l) : null), backgroundColor:'#31c48d', borderRadius:6, maxBarThickness:38},
          {label:'Zaman dilimi isabet %', data:labels.map(l => timeframeMap.has(l) ? timeframeMap.get(l) : null), backgroundColor:'#6ca7ff', borderRadius:6, maxBarThickness:38},
        ],
      },
      options:{ responsive:true, plugins:{ legend:{ labels:{ color:'#e7eefb', usePointStyle:true, boxWidth:8 } }, tooltip:{callbacks:{label:(c)=>c.dataset.label+': %'+c.parsed.y}} },
        scales:{ y:{ beginAtZero:true, max:100, ticks:{callback:(v)=>v+'%'}, grid:{ color:'rgba(255,255,255,.06)' } }, x:{ grid:{ display:false } } } }
    });
  } else if (barChart){ barChart.destroy(); barChart = null; }

    document.getElementById('watchlist').innerHTML = d.watchlist.length ? d.watchlist.map(w => `<tr>
      <td><b>${esc(w.symbol)}</b></td><td class="muted">${esc(w.name || '')}</td>
      <td class="muted">${esc(w.profiles || '')} / ${esc(w.timeframes || '')}</td>
      <td style="text-align:right;white-space:nowrap">
        <button class="btn good small" onclick='analyzeWatchlistItem(${JSON.stringify(w).replace(/'/g,"&#39;")})'>▶ Analiz</button>
        <button class="btn alt small" onclick='editWatchlist(${JSON.stringify(w).replace(/'/g,"&#39;")})'>Seç</button>
        <button class="btn alt small" title="Favorilerden kaldır" onclick='removeWatchlistItem("${esc(w.symbol)}")'>🗑</button>
      </td></tr>`).join('') : '<tr><td colspan="4" class="empty">Favori listesi boş — bir sembol seçip "Favorilere Ekle" butonuna basın.</td></tr>';

    // Per-symbol scorecard: hit-rate bars over the last 30 days.
    const perf = (d.by_symbol || []).filter(x => x.total > 0);
    document.getElementById('symperf').innerHTML = perf.length ? perf.map(x => {
      const pct = x.total ? Math.round(100 * x.hits / x.total) : 0;
      const col = pct >= 55 ? 'var(--green)' : (pct <= 45 ? 'var(--red)' : 'var(--amber)');
      return `<tr>
        <td><b>${esc(x.symbol)}</b></td><td>${x.total}</td><td>${x.hits}</td>
        <td><div class="perfbar"><div class="track"><i style="width:${pct}%;background:${col}"></i></div><span style="color:${col};font-weight:700">%${pct}</span></div></td>
        <td class="muted">${x.avg_confidence == null ? '—' : '%' + Math.round(x.avg_confidence * 100)}</td>
      </tr>`;
    }).join('') : '<tr><td colspan="5" class="empty">Henüz sonuçlanmış tahmin yok.</td></tr>';
}

function analyzeWatchlistItem(w){
  editWatchlist(w);       // selects it (and switches to the Analiz tab)
  analyze();
}

async function removeWatchlistItem(symbol){
  await fetch('/api/watchlist/' + encodeURIComponent(symbol), {method:'DELETE'});
  loadDashboard();
}

let compareChart = null;
function renderComparison(results){
  const card = document.getElementById('compareCard');
  if (!results || results.length < 2){ card.classList.add('is-hidden'); return; }
  card.classList.remove('is-hidden');
  compareChart = chartOrUpdate(compareChart, document.getElementById('compareChartCanvas'), {
    type: 'bar',
    data: {
      labels: results.map(p => p.symbol),
      datasets: [{
        label: 'Final Skor',
        data: results.map(p => p.final_score),
        backgroundColor: results.map(p => p.final_score > 0.15 ? '#31c48d' : (p.final_score < -0.15 ? '#ff6b6b' : '#8ea4c7')),
        borderRadius: 6, maxBarThickness: 46,
      }],
    },
    options: { responsive:true, plugins:{legend:{display:false}, tooltip:{callbacks:{label:(c)=>'Final skor: '+c.parsed.y.toFixed(2)}}},
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

function scoreBadge(label, score, confidence, confLabel, confTitle){
  const clamped = Math.max(-1, Math.min(1, score));
  const barPct = Math.round(((clamped + 1) / 2) * 100);
  const color = score > 0.15 ? 'var(--green)' : (score < -0.15 ? 'var(--red)' : 'var(--muted)');
  const titleAttr = confTitle ? ` title="${esc(confTitle)}"` : '';
  const confRow = confidence === undefined || confidence === null ? '' :
    `<div class="muted" style="font-size:11px"${titleAttr}>${esc(confLabel || 'güven')} %${Math.round(confidence * 100)}</div>`;
  return `<div class="score-badge">
    <div class="muted" style="font-size:11px">${esc(label)}</div>
    <div class="num" style="color:${color}">${score >= 0 ? '+' : ''}${score.toFixed(2)}</div>
    <div class="score-bar"><i style="width:${barPct}%;background:${color}"></i></div>
    ${confRow}
  </div>`;
}

function scoreColor(v){ return v > 0.05 ? 'var(--green)' : (v < -0.05 ? 'var(--red)' : 'var(--muted)'); }
function softDir(v, band){ return v > band ? 'UP' : (v < -band ? 'DOWN' : 'NEUTRAL'); }
function rsiLabel(v){ return v == null ? '—' : (v >= 70 ? 'Aşırı alım' : (v <= 30 ? 'Aşırı satım' : 'Nötr')); }
function volLabel(pct){ return pct == null ? '—' : (pct >= 4 ? 'Yüksek' : (pct >= 2 ? 'Orta' : 'Düşük')); }

function indCategory(name){
  const n = (name || '').toLowerCase();
  if (n.includes('rsi') || n.includes('macd')) return 'Momentum';
  if (n.includes('hacim') || n.includes('volume')) return 'Hacim';
  if (n.includes('bollinger')) return 'Volatilite';
  return 'Trend';
}
const CATEGORY_ICON = {Trend:'📈', Momentum:'⚡', Hacim:'🔊', Volatilite:'🎯'};

function renderIndicatorCards(indicators){
  if (!indicators || !indicators.length){
    return '<div class="muted">Bu sembol için yeterli fiyat geçmişi olmadığından teknik göstergeler hesaplanamadı (en az 50 işlem günü gerekir).</div>';
  }
  return `<div class="ind-cards">${indicators.map(ind => `
    <div class="ind-card ${dirClass(ind.direction)}">
      <div class="top">
        <span class="nm">${CATEGORY_ICON[indCategory(ind.name)] || ''} ${esc(ind.name)}</span>
        <span class="val ${dirClass(ind.direction)}">${dirArrow(ind.direction)} ${esc(ind.value)}</span>
      </div>
      <div class="exp">${esc(ind.explanation)}</div>
      <div class="wbar" title="Teknik skordaki ağırlığı %${ind.weight_pct}"><i style="width:${ind.weight_pct}%"></i></div>
    </div>`).join('')}</div>`;
}

function techSummaryText(p){
  const inds = p.technical_indicators || [];
  if (!inds.length) return '';
  const up = inds.filter(i => i.direction === 'UP').length;
  const down = inds.filter(i => i.direction === 'DOWN').length;
  const neu = inds.length - up - down;
  const lean = softDir(p.technical_score, 0.05);
  const sign = p.technical_score >= 0 ? 'UP' : 'DOWN';
  const byCat = {};
  inds.forEach(i => { if (i.direction === sign){ const c = indCategory(i.name); byCat[c] = (byCat[c] || 0) + i.weight_pct; } });
  const top = Object.entries(byCat).sort((a, b) => b[1] - a[1])[0];
  const topTxt = top ? ` En güçlü katkı: <b>${esc(top[0])}</b>.` : '';
  return `<div class="tech-summary">Teknik skor <b style="color:${scoreColor(p.technical_score)}">${p.technical_score >= 0 ? '+' : ''}${p.technical_score.toFixed(2)}</b> → göstergeler ağırlıklı olarak <b>${dirLabel(lean)}</b> yönünde
    (${up} yukarı · ${down} aşağı · ${neu} nötr).${topTxt}</div>`;
}

function verdictBanner(p){
  const d = p.final_direction;
  const conf = Math.round((p.final_confidence || 0) * 100);
  let txt;
  if (p.profile === 'learned'){
    // In learned mode the score IS the model's output: P(up) = (score+1)/2.
    const clamped = Math.max(-1, Math.min(1, p.final_score));
    const pUp = Math.round(((clamped + 1) / 2) * 100);
    // Look up this call's conviction band in the model's own out-of-sample
    // track record, so "how much should I trust THIS call" has a real number.
    let track = '';
    const sel = (modelMeta && modelMeta.selective) || [];
    if (sel.length){
      const conviction = Math.abs(clamped) / 2;   // |score|/2 == |P-0.5|
      const band = sel.filter(b => conviction >= b.min_conviction).sort((a, b) => b.min_conviction - a.min_conviction)[0];
      if (band) track = ` Bu güç seviyesindeki geçmiş çağrılarda (out-of-sample) isabet <b>%${(band.accuracy * 100).toFixed(1)}</b> (kapsam %${(band.coverage * 100).toFixed(1)}).`;
    }
    txt = `🤖 <b>Öğrenen model tahmini (≈3 ay):</b> ${dirLabel(d)} — yükseliş olasılığı <b>%${pUp}</b>.${track}
      <span class="muted">Haber/teknik alt-skorları aşağıda; model bunları birlikte öğrendi.</span>`;
  } else {
    const nd = softDir(p.news_score, 0.1);
    const td = softDir(p.technical_score, 0.1);
    const agree = nd !== 'NEUTRAL' && nd === td;
    const conflict = nd !== 'NEUTRAL' && td !== 'NEUTRAL' && nd !== td;
    const note = agree ? 'Haber ve teknik taraf aynı yönde — sinyal güçlü.'
      : (conflict ? 'Haber ve teknik taraf ayrışıyor — temkinli olun.'
      : 'Taraflardan biri nötr — sinyal ılımlı.');
    txt = `Haber tarafı <b>${dirLabel(nd)}</b>, teknik taraf <b>${dirLabel(td)}</b>. ${note}`;
    // The score below is SIGNAL STRENGTH, not "how likely this call is correct" —
    // spell that out explicitly whenever the final call lands on NEUTRAL, since
    // that's exactly when a high number next to "NEUTRAL" looks contradictory.
    if (d === 'NEUTRAL'){
      txt += conflict
        ? `<div class="muted" style="font-size:12px;margin-top:6px">ℹ️ Sinyal Gücü yüksek görünse de bu "nötr karar kesin doğru" demek değildir — her iki taraf da güçlü konuşuyor ama <b>zıt yönlerde</b>, birbirini götürüyor. Yön belirsiz demektir.</div>`
        : `<div class="muted" style="font-size:12px;margin-top:6px">ℹ️ Sinyaller zayıf/belirsiz olduğu için final skor nötr bölgede kaldı.</div>`;
    }
  }
  const confLabel = p.profile === 'learned' ? 'Model Güveni (olasılık sapması)' : 'Sinyal Gücü';
  const confTitle = p.profile === 'learned'
    ? 'Modelin %50 (yazı-tura) tahmininden ne kadar uzaklaştığı. Yön tahmininin isabet garantisi değildir.'
    : 'Yön tahmininin doğru olma ihtimali DEĞİLDİR. Haber ve teknik sinyallerin birlikte ne kadar güçlü/net konuştuğunu ölçer.';
  return `<div class="verdict ${dirClass(d)}">
    <div class="big ${dirClass(d)}">${dirArrow(d)} ${dirLabel(d)}</div>
    <div class="txt">${txt}</div>
    <div class="conf">
      <div class="conf-label" title="${esc(confTitle)}">${confLabel}: %${conf}</div>
      <div class="conf-bar"><i style="width:${conf}%"></i></div>
    </div>
  </div>`;
}

function statStrip(sym, sm){
  if (!sm || sm.last == null) return '';
  const ch = sm.change_pct || 0;
  const cell = (k, v, extra = '') => `<div class="cell"><div class="k">${k}</div><div class="v" ${extra}>${v}</div></div>`;
  return cell('Güncel Fiyat', formatPrice(sym, sm.last))
    + cell('Değişim (son bar)', `${ch >= 0 ? '▲' : '▼'} ${Math.abs(ch).toFixed(2)}%`, `style="color:${scoreColor(ch)}"`)
    + cell('Dönem En Yüksek', formatPrice(sym, sm.period_high))
    + cell('Dönem En Düşük', formatPrice(sym, sm.period_low))
    + cell('RSI (14)', `${sm.rsi == null ? '—' : sm.rsi.toFixed(0)} · ${rsiLabel(sm.rsi)}`)
    + cell('Volatilite (ATR)', `${sm.atr_pct == null ? '—' : sm.atr_pct.toFixed(1) + '%'} · ${volLabel(sm.atr_pct)}`);
}

function positionBar(sym, sm){
  if (!sm || sm.last == null) return '';
  const pos = Math.max(0, Math.min(100, sm.position_pct));
  return `<div class="section-title" style="margin:6px 0 4px">📍 Fiyatın Dönem Aralığındaki Konumu</div>
    <div class="pos-track"><div class="marker" style="left:${pos}%"></div></div>
    <div class="pos-ends"><span>En düşük ${formatPrice(sym, sm.period_low)}</span><span>%${pos.toFixed(0)}</span><span>En yüksek ${formatPrice(sym, sm.period_high)}</span></div>
    <div class="muted" style="font-size:12px;margin-top:8px">🟢 Yakın destek: <b>${formatPrice(sym, sm.support)}</b> &nbsp;·&nbsp; 🔴 Yakın direnç: <b>${formatPrice(sym, sm.resistance)}</b></div>`;
}

let currentDetailIdx = -1;

function highlightResultRow(idx){
  document.querySelectorAll('#results .rcard').forEach(el => el.classList.remove('active'));
  const row = document.getElementById('resrow-' + idx);
  if (row) row.classList.add('active');
}

let detailToken = 0;

function openDetail(idx, scroll){
  const p = lastResults[idx];
  if (!p) return;
  currentDetailIdx = idx;
  highlightResultRow(idx);
  document.getElementById('detailSwitch').textContent =
    lastResults.length > 1 ? 'Tablodan başka bir karta tıklayarak değiştirebilirsiniz' : '';
  showDetailFor(p, scroll);
}

// Open a stored (history) prediction: same panel, live chart/news/indicators.
function openHistoryDetail(row){
  currentDetailIdx = -1;
  highlightResultRow(-1);
  showTab('analiz');
  document.getElementById('detailSwitch').textContent = 'Geçmiş tahmin — grafik ve haberler güncel veriyle yenilenir';
  showDetailFor({
    symbol: row.symbol, name: '', timeframe: row.timeframe || '1d', profile: row.profile || 'balanced',
    final_score: row.final_score ?? 0, final_direction: row.final_direction || 'NEUTRAL',
    final_confidence: row.final_confidence ?? 0, news_score: row.news_score ?? 0,
    news_confidence: row.news_confidence ?? 0, news_rationale: row.news_rationale || '',
    technical_score: row.technical_score ?? 0, technical_indicators: [],
    price_at_prediction: row.price_at_prediction ?? null, news_sources: row.news_sources || 'google',
  }, true);
}

async function showDetailFor(p, scroll){
  const panel = document.getElementById('detailPanel');
  panel.classList.remove('is-hidden');
  const token = ++detailToken;

  // Instant (no-fetch) parts render immediately.
  document.getElementById('detailTitle').textContent = `📊 ${p.symbol}${p.name ? ' — ' + p.name : ''}`;
  document.getElementById('detailSub').textContent =
    `${p.timeframe} · ${p.profile} · analiz anı fiyatı ${formatPrice(p.symbol, p.price_at_prediction)}`;
  document.getElementById('detailVerdict').innerHTML = verdictBanner(p);
  document.getElementById('detailScores').innerHTML =
    (p.profile === 'learned'
      ? scoreBadge('🤖 Model Skoru', p.final_score, p.final_confidence, 'model güveni',
          'Modelin %50 (yazı-tura) tahmininden ne kadar uzaklaştığı — yön tahmininin isabet garantisi değildir.')
      : scoreBadge('Final Skor', p.final_score, p.final_confidence, 'sinyal gücü',
          'Bu, yön tahmininin doğru olma ihtimali DEĞİLDİR. Haber ve teknik sinyallerin birlikte ne kadar güçlü/net konuştuğunu gösterir. Sinyaller çelişirse (biri yükseliş biri düşüş derse) final yön nötr olabilir ama bu değer yine de yüksek görünebilir.')) +
    scoreBadge('Haber Skoru (AI)', p.news_score, p.news_confidence, 'AI güveni',
      'Yapay zekânın KENDİ haber okumasına ne kadar güvendiği — haberin fiyatı gerçekten hareket ettireceğinin garantisi değildir.') +
    scoreBadge('Teknik Skor', p.technical_score, null);
  document.getElementById('detailTechSummary').innerHTML = techSummaryText(p);
  document.getElementById('detailIndicators').innerHTML = renderIndicatorCards(p.technical_indicators);
  document.getElementById('detailNewsRationale').textContent = p.news_rationale || 'Yorum yok.';
  document.getElementById('detailSummary').innerHTML = '';
  document.getElementById('detailPosition').innerHTML = '';
  document.getElementById('detailLevels').innerHTML = '';
  document.getElementById('detailPlan').innerHTML = '';
  document.getElementById('detailNews').innerHTML = '<div class="muted">Haberler yükleniyor...</div>';

  if (scroll) panel.scrollIntoView({behavior:'smooth', block:'start'});

  const sources = p.news_sources || 'google';
  const newsUrl = `/api/news?symbol=${encodeURIComponent(p.symbol)}&name=${encodeURIComponent(p.name || p.symbol)}&sources=${encodeURIComponent(sources)}`;
  const chartUrl = `/api/chart?symbol=${encodeURIComponent(p.symbol)}&timeframe=${encodeURIComponent(p.timeframe || '1d')}`;

  const [newsResp, chartResp] = await Promise.all([fetch(newsUrl), fetch(chartUrl)]);
  const news = await newsResp.json();
  const chartData = await chartResp.json();
  if (token !== detailToken) return;  // a newer open won the race

  const sm = chartData.summary || {};
  document.getElementById('detailSummary').innerHTML = statStrip(p.symbol, sm);
  document.getElementById('detailPosition').innerHTML = positionBar(p.symbol, sm);

  // History rows carry no indicator breakdown — fill it from the fresh calc.
  if ((!p.technical_indicators || !p.technical_indicators.length) && (chartData.technical_indicators || []).length){
    document.getElementById('detailIndicators').innerHTML = renderIndicatorCards(chartData.technical_indicators);
    document.getElementById('detailTechSummary').innerHTML = techSummaryText({
      technical_score: chartData.technical_score ?? 0, final_direction: p.final_direction,
      technical_indicators: chartData.technical_indicators,
    });
  }

  document.getElementById('detailNews').innerHTML = news.length ? news.map(a => `
    <div class="news-item">
      <a href="${esc(a.url)}" target="_blank" rel="noopener">${esc(a.title)}</a>
      <div class="news-meta">${esc(a.source)} · ${new Date(a.published_ts).toLocaleString('tr-TR')}</div>
    </div>`).join('') : '<div class="muted">Bu sembol için haber bulunamadı.</div>';

  lastChartData = chartData;
  lastChartSym = p.symbol;
  lastChartTf = p.timeframe || '1d';
  const tfEl = document.querySelector(`input[name="charttf"][value="${lastChartTf}"]`);
  if (tfEl) tfEl.checked = true;
  renderDetailChart();
  renderLevels(p.symbol, sm);
  renderTradePlan(p.symbol, sm);
}

// ---- Detail price chart: line/candle + optional pivot/Fibonacci overlays ----
let lastChartData = null;
let lastChartSym = null;

function chartTypeIsCandle(){
  const el = document.querySelector('input[name="charttype"]:checked');
  return el && el.value === 'candle';
}
// Individually toggleable overlays (SMA/EMA/Bollinger/VWAP/Volume/S-R/Pivot/Fib).
function indOn(name){
  const el = document.querySelector(`#indicatorToggles [data-ind="${name}"]`);
  return el ? el.checked : false;
}
function priceScaleMode(){
  const el = document.querySelector('input[name="pricescale"]:checked');
  return (el && el.value === 'log') ? LightweightCharts.PriceScaleMode.Logarithmic : LightweightCharts.PriceScaleMode.Normal;
}

// Persist chart type / scale / indicator toggles across symbols and reloads —
// a trader shouldn't have to re-enable their preferred overlays every time.
const CHART_PREFS_KEY = 'tradeview_chart_prefs_v1';
function saveChartPrefs(){
  const prefs = {
    type: document.querySelector('input[name="charttype"]:checked')?.value || 'line',
    scale: document.querySelector('input[name="pricescale"]:checked')?.value || 'linear',
    ind: {},
  };
  document.querySelectorAll('#indicatorToggles [data-ind]').forEach(el => { prefs.ind[el.dataset.ind] = el.checked; });
  try { localStorage.setItem(CHART_PREFS_KEY, JSON.stringify(prefs)); } catch (e) {}
}
function loadChartPrefs(){
  let prefs = null;
  try { prefs = JSON.parse(localStorage.getItem(CHART_PREFS_KEY) || 'null'); } catch (e) {}
  if (!prefs) return;
  const typeEl = document.querySelector(`input[name="charttype"][value="${prefs.type}"]`);
  if (typeEl) typeEl.checked = true;
  const scaleEl = document.querySelector(`input[name="pricescale"][value="${prefs.scale}"]`);
  if (scaleEl) scaleEl.checked = true;
  document.querySelectorAll('#indicatorToggles [data-ind]').forEach(el => {
    if (prefs.ind && Object.prototype.hasOwnProperty.call(prefs.ind, el.dataset.ind)) el.checked = prefs.ind[el.dataset.ind];
  });
}
function onChartPrefChange(){ saveChartPrefs(); updateVwapAvailability(); renderDetailChart(); }
function redrawDetailChart(){ renderDetailChart(); }

// TradingView Lightweight Charts — crisp, professional, big.
let lwChart = null, lwMain = null;
let lwTimeIndex = new Map();   // stringified lwTime -> bar index, for the crosshair legend

let lastChartTf = '1d';
function isIntradayTf(tf){ return tf === '30m' || tf === '1h'; }
// VWAP only means something within a trading session — grey it out on daily+.
function updateVwapAvailability(){
  const el = document.querySelector('#indicatorToggles [data-ind="vwap"]');
  if (!el) return;
  const ok = isIntradayTf(lastChartTf);
  el.disabled = !ok;
  const label = el.closest('label');
  if (label) label.title = ok ? '' : 'VWAP sadece gün içi (30 dk / 1 saat) zaman diliminde anlamlıdır.';
}
// Daily+ bars use a clean business-day axis; intraday uses timestamps + time.
function lwTime(iso){
  const d = new Date(iso);
  if (isIntradayTf(lastChartTf)) return Math.floor(d.getTime() / 1000);
  return {year: d.getUTCFullYear(), month: d.getUTCMonth() + 1, day: d.getUTCDate()};
}
function lwLineData(dates, arr){
  const out = [];
  for (let i = 0; i < arr.length; i++){ if (arr[i] != null) out.push({time: lwTime(dates[i]), value: arr[i]}); }
  return out;
}
function lwThemeOpts(el, height){
  return {
    width: el.clientWidth, height,
    layout: {background: {type: 'solid', color: 'transparent'}, textColor: '#a9bcda', fontSize: 12, fontFamily: 'Inter,Segoe UI,sans-serif'},
    grid: {vertLines: {color: 'rgba(255,255,255,.05)'}, horzLines: {color: 'rgba(255,255,255,.08)'}},
    crosshair: {mode: LightweightCharts.CrosshairMode.Normal},
    rightPriceScale: {borderColor: 'rgba(255,255,255,.14)', scaleMargins: {top: 0.08, bottom: 0.08}, mode: priceScaleMode()},
    timeScale: {borderColor: 'rgba(255,255,255,.14)', timeVisible: isIntradayTf(lastChartTf), secondsVisible: false, rightOffset: 3, minBarSpacing: 3},
  };
}
// Show the most recent ~`show` bars by default (readable), full history scrollable.
function setLwView(chart, n, show){
  if (n > show) chart.timeScale().setVisibleLogicalRange({from: n - show, to: n - 1 + 3});
  else chart.timeScale().fitContent();
}

function renderDetailChart(){
  if (!lastChartData) return;
  updateVwapAvailability();
  const cd = lastChartData, sm = cd.summary || {}, dates = cd.dates || [];
  const priceEl = document.getElementById('lwPrice');
  if (typeof LightweightCharts === 'undefined' || !priceEl){
    if (priceEl) priceEl.innerHTML = '<div class="chart-empty">Grafik kütüphanesi yüklenemedi.</div>';
    return;
  }
  if (lwChart){ lwChart.remove(); lwChart = null; }
  lwTimeIndex = new Map();
  dates.forEach((d, i) => lwTimeIndex.set(JSON.stringify(lwTime(d)), i));

  // --- price pane ---
  lwChart = LightweightCharts.createChart(priceEl, lwThemeOpts(priceEl, 460));
  if (chartTypeIsCandle() && cd.open && cd.high && cd.low){
    lwMain = lwChart.addCandlestickSeries({upColor: '#31c48d', downColor: '#ff6b6b', borderVisible: false, wickUpColor: '#31c48d', wickDownColor: '#ff6b6b'});
    lwMain.setData(dates.map((d, i) => ({time: lwTime(d), open: cd.open[i], high: cd.high[i], low: cd.low[i], close: cd.close[i]})));
  } else {
    lwMain = lwChart.addAreaSeries({lineColor: '#6ca7ff', topColor: 'rgba(108,167,255,.35)', bottomColor: 'rgba(108,167,255,0)', lineWidth: 2});
    lwMain.setData(lwLineData(dates, cd.close));
  }
  const overlay = (arr, color, w, dashed) => {
    if (!arr) return;
    const s = lwChart.addLineSeries({color, lineWidth: w, lineStyle: dashed ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false});
    s.setData(lwLineData(dates, arr));
  };
  if (indOn('sma50')) overlay(cd.sma50, '#f4b942', 2);
  if (indOn('sma200')) overlay(cd.sma200, '#ff6b6b', 2);
  if (indOn('ema20')) overlay(cd.ema20, '#31c48d', 1);
  if (indOn('bb')){
    overlay(cd.bb_upper, 'rgba(142,164,199,.5)', 1);
    overlay(cd.bb_lower, 'rgba(142,164,199,.5)', 1);
  }
  if (indOn('vwap') && isIntradayTf(lastChartTf)) overlay(cd.vwap, '#22d3ee', 2, true);
  // volume histogram, tucked into the bottom 18%
  if (indOn('volume') && cd.volume){
    const vol = lwChart.addHistogramSeries({priceScaleId: 'vol', priceFormat: {type: 'volume'}, priceLineVisible: false, lastValueVisible: false});
    lwChart.priceScale('vol').applyOptions({scaleMargins: {top: 0.82, bottom: 0}});
    vol.setData(dates.map((d, i) => ({time: lwTime(d), value: cd.volume[i] || 0, color: cd.close[i] >= cd.open[i] ? 'rgba(49,196,141,.35)' : 'rgba(255,107,107,.35)'})));
  }
  // level lines
  const priceLine = (price, color, title, dashed) => {
    if (price == null || !isFinite(price)) return;
    lwMain.createPriceLine({price, color, lineWidth: 1, lineStyle: dashed ? LightweightCharts.LineStyle.Dashed : LightweightCharts.LineStyle.Solid, axisLabelVisible: true, title});
  };
  if (indOn('sr')){
    priceLine(sm.support, '#31c48d', 'Destek');
    priceLine(sm.resistance, '#ff6b6b', 'Direnç');
  }
  if (indOn('pivot') && sm.pivot){
    priceLine(sm.pivot.p, 'rgba(108,167,255,.75)', 'P', true);
    priceLine(sm.pivot.r1, 'rgba(108,167,255,.5)', 'R1', true);
    priceLine(sm.pivot.s1, 'rgba(108,167,255,.5)', 'S1', true);
  }
  if (indOn('fib') && sm.fib){
    priceLine(sm.fib['38.2'], 'rgba(196,132,252,.6)', 'Fib 38', true);
    priceLine(sm.fib['50'], 'rgba(196,132,252,.6)', 'Fib 50', true);
    priceLine(sm.fib['61.8'], 'rgba(196,132,252,.6)', 'Fib 62', true);
  }
  const n = dates.length, show = Math.min(n, 130);
  setLwView(lwChart, n, show);

  // Crosshair legend: OHLC + %change + volume, defaulting to the latest bar.
  lwChart.subscribeCrosshairMove((param) => {
    let i = (param && param.time != null) ? lwTimeIndex.get(JSON.stringify(param.time)) : undefined;
    if (i == null) i = cd.close.length - 1;
    renderChartLegend(cd, i);
  });
  renderChartLegend(cd, cd.close.length - 1);
}
function renderChartLegend(cd, i){
  const box = document.getElementById('chartLegend');
  if (!box) return;
  if (i == null || cd.close[i] == null){ box.innerHTML = ''; return; }
  const sym = lastChartSym, o = cd.open[i], h = cd.high[i], l = cd.low[i], c = cd.close[i];
  const v = (cd.volume && cd.volume[i] != null) ? cd.volume[i] : null;
  const prevClose = i > 0 && cd.close[i - 1] != null ? cd.close[i - 1] : c;
  const chg = prevClose ? ((c / prevClose - 1) * 100) : 0;
  const col = chg >= 0 ? 'var(--green)' : 'var(--red)';
  const dateLbl = new Date(cd.dates[i]).toLocaleString('tr-TR', isIntradayTf(lastChartTf)
    ? {day:'2-digit', month:'2-digit', hour:'2-digit', minute:'2-digit'} : {day:'2-digit', month:'2-digit', year:'numeric'});
  box.innerHTML = `<b>${esc(sym || '')}</b><span class="muted">${dateLbl}</span>
    <span>A <b>${formatPrice(sym, o)}</b></span>
    <span>Y <b>${formatPrice(sym, h)}</b></span>
    <span>D <b>${formatPrice(sym, l)}</b></span>
    <span style="color:${col}">K <b>${formatPrice(sym, c)}</b> (${chg >= 0 ? '+' : ''}${chg.toFixed(2)}%)</span>
    ${v != null ? `<span class="muted">Hacim ${Math.round(v).toLocaleString('tr-TR')}</span>` : ''}`;
}
function resizeLwCharts(){
  const p = document.getElementById('lwPrice');
  if (lwChart && p && p.clientWidth) lwChart.applyOptions({width: p.clientWidth});
}
// Flip the chart between timeframes without re-running the (Groq-costing) full
// analysis — a quick 1D/1W/1M/1h/30m switch is table-stakes for a trader.
async function switchChartTimeframe(tf){
  if (!lastChartSym) return;
  lastChartTf = tf;
  updateVwapAvailability();
  const r = await fetch(`/api/chart?symbol=${encodeURIComponent(lastChartSym)}&timeframe=${encodeURIComponent(tf)}`);
  if (!r.ok) return;
  const chartData = await r.json();
  lastChartData = chartData;
  renderDetailChart();
  renderLevels(lastChartSym, chartData.summary || {});
  renderTradePlan(lastChartSym, chartData.summary || {});
}
window.addEventListener('resize', resizeLwCharts);

function lvlCell(k, v, sym){ return `<div class="lv"><div class="k">${esc(k)}</div><div class="v">${formatPrice(sym, v)}</div></div>`; }
function renderLevels(sym, sm){
  const box = document.getElementById('detailLevels');
  if (!sm || (!sm.pivot && !sm.fib)){ box.innerHTML = '<div class="muted">Seviye verisi yok.</div>'; return; }
  let html = '';
  if (sm.pivot){
    const p = sm.pivot;
    html += '<div class="muted" style="font-size:12px;margin:2px 0 4px">Pivot noktaları (destek S · pivot P · direnç R)</div><div class="levels">'
      + lvlCell('S2', p.s2, sym) + lvlCell('S1', p.s1, sym) + lvlCell('P', p.p, sym)
      + lvlCell('R1', p.r1, sym) + lvlCell('R2', p.r2, sym) + '<div class="lv"></div></div>';
  }
  if (sm.fib){
    const f = sm.fib;
    html += '<div class="muted" style="font-size:12px;margin:8px 0 4px">Fibonacci geri çekilme</div><div class="levels">'
      + lvlCell('%0', f['0'], sym) + lvlCell('%23.6', f['23.6'], sym) + lvlCell('%38.2', f['38.2'], sym)
      + lvlCell('%50', f['50'], sym) + lvlCell('%61.8', f['61.8'], sym) + lvlCell('%100', f['100'], sym) + '</div>';
  }
  box.innerHTML = html;
}

// Turn the technical levels into a concrete "buy below / sell above" ladder.
function renderTradePlan(sym, sm){
  const box = document.getElementById('detailPlan');
  if (!sm || sm.last == null){ box.innerHTML = '<div class="muted">Plan için yeterli veri yok.</div>'; return; }
  const last = sm.last;
  const levels = [];
  const seen = new Set();
  const push = (price, label) => {
    if (price == null || !isFinite(price)) return;
    const key = Number(price).toFixed(2);
    if (seen.has(key)) return;
    seen.add(key);
    levels.push({price: Number(price), label});
  };
  push(sm.resistance, 'Direnç'); push(sm.support, 'Destek');
  push(sm.period_high, 'Dönem zirvesi'); push(sm.period_low, 'Dönem dibi');
  if (sm.pivot){ push(sm.pivot.r2, 'Pivot R2'); push(sm.pivot.r1, 'Pivot R1'); push(sm.pivot.p, 'Pivot P'); push(sm.pivot.s1, 'Pivot S1'); push(sm.pivot.s2, 'Pivot S2'); }
  if (sm.fib){ ['0', '23.6', '38.2', '50', '61.8', '100'].forEach(k => push(sm.fib[k], 'Fib %' + k)); }

  const dist = (pr) => (pr / last - 1) * 100;
  const sells = levels.filter(x => x.price > last).sort((a, b) => a.price - b.price).slice(0, 5);
  const buys = levels.filter(x => x.price <= last).sort((a, b) => b.price - a.price).slice(0, 5);
  const row = (tip, cls, x) => `<tr><td class="${cls}">${tip}</td><td>${formatPrice(sym, x.price)}</td><td class="muted">${esc(x.label)}</td><td style="color:${dist(x.price) >= 0 ? 'var(--green)' : 'var(--red)'}">${dist(x.price) >= 0 ? '+' : ''}${dist(x.price).toFixed(1)}%</td></tr>`;
  const sellRows = sells.slice().sort((a, b) => b.price - a.price).map(x => row('SAT', 'tag-sell', x)).join('');
  const buyRows = buys.map(x => row('AL', 'tag-buy', x)).join('');
  const curRow = `<tr class="cur"><td>◆ ŞİMDİ</td><td>${formatPrice(sym, last)}</td><td class="muted">güncel fiyat</td><td>—</td></tr>`;
  const stop = sm.atr ? last - 1.5 * sm.atr : null;
  const stopNote = stop ? `<div class="muted" style="font-size:12px;margin-top:8px">🛑 Önerilen stop (uzun pozisyon, 1.5×ATR): <b>${formatPrice(sym, stop)}</b> (${(dist(stop)).toFixed(1)}%)</div>` : '';
  box.innerHTML = `<table class="plan"><thead><tr><th>Tip</th><th>Seviye</th><th>Kaynak</th><th>Uzaklık</th></tr></thead>
    <tbody>${sellRows}${curRow}${buyRows}</tbody></table>${stopNote}
    <div class="muted" style="font-size:11px;margin-top:8px">🟢 AL = fiyatın altındaki destek/geri-çekilme bölgeleri · 🔴 SAT = üstteki direnç/hedefler. Teknik referanslardır, yatırım tavsiyesi değildir.</div>`;
}

// Show the learned model's honest report card.
let modelMeta = null;   // stashed for the learned-mode verdict banner
async function loadModelInfo(){
  const box = document.getElementById('modelBody');
  let d = {available: false};
  try { const r = await fetch('/api/model'); if (r.ok) d = await r.json(); } catch (e) {}
  if (!d.available){
    box.innerHTML = '<div class="muted">Henüz eğitilmiş model yok. Sunucuda <code>python main.py train --universe all</code> ile eğitilir; "Öğrenen model" profili o zaman devreye girer.</div>';
    return;
  }
  const m = d.meta || {};
  modelMeta = m;
  const acc = (m.accuracy != null) ? (m.accuracy * 100).toFixed(1) : '—';
  const base = (m.baseline_accuracy != null) ? (m.baseline_accuracy * 100).toFixed(1) : '—';
  const auc = (m.auc != null) ? m.auc.toFixed(3) : '—';
  const edge = (m.accuracy != null && m.baseline_accuracy != null) ? '+' + ((m.accuracy - m.baseline_accuracy) * 100).toFixed(1) + ' puan' : '—';
  const when = m.trained_at ? new Date(m.trained_at).toLocaleString('tr-TR') : '—';
  const cell = (k, v, extra = '') => `<div class="cell"><div class="k">${k}</div><div class="v" ${extra}>${v}</div></div>`;
  const strip = '<div class="statstrip" style="grid-template-columns:repeat(auto-fit,minmax(140px,1fr))">'
    + cell('Doğruluk (test)', '%' + acc, 'style="color:var(--green)"')
    + cell('Baseline (trend)', '%' + base)
    + cell('Taban oran (hep-Yükseliş)', m.base_rate != null ? '%' + (m.base_rate * 100).toFixed(1) : '—')
    + cell('AUC (ayırt edicilik)', auc)
    + cell('Test örneği', (m.n_test || 0).toLocaleString('tr-TR'))
    + cell('Ufuk', (m.horizon || '—') + ' bar (~3 ay)')
    + '</div>';
  // The professional part: accuracy when acting only on stronger signals.
  const sel = (m.selective || []).filter(b => b.min_conviction > 0);
  const selGrid = sel.length ? '<div class="muted" style="font-size:12px;margin-top:14px">🎯 <b>Seçici mod:</b> model yalnızca güçlü sinyallerde kullanıldığında (out-of-sample):</div>'
    + '<div class="sel-grid">' + sel.map(b => `
      <div class="sel-cell ${b.accuracy >= 0.65 ? 'hot' : ''}">
        <div class="t">Sinyal gücü ≥ ${(b.min_conviction * 100).toFixed(0)}</div>
        <div class="a" style="color:${b.accuracy >= 0.65 ? 'var(--green)' : 'var(--text)'}">%${(b.accuracy * 100).toFixed(1)}</div>
        <div class="c">kapsam %${(b.coverage * 100).toFixed(1)} · n=${b.n.toLocaleString('tr-TR')}</div>
      </div>`).join('') + '</div>' : '';
  const w = d.weights || {};
  const names = Object.keys(w);
  const maxW = Math.max(0.0001, ...names.map(n => Math.abs(w[n])));
  const bars = names.sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).map(n => {
    const val = w[n];
    const half = Math.abs(val) / maxW * 50;
    const seg = val >= 0
      ? `<i style="left:50%;width:${half}%;background:var(--green)"></i>`
      : `<i style="left:${50 - half}%;width:${half}%;background:var(--red)"></i>`;
    return `<div class="wrow"><div class="wl">${esc(n)}</div><div class="wt"><div class="z"></div>${seg}</div><div class="wv" style="color:${val >= 0 ? 'var(--green)' : 'var(--red)'}">${val >= 0 ? '+' : ''}${val.toFixed(3)}</div></div>`;
  }).join('');
  box.innerHTML = strip
    + selGrid
    + `<div class="muted" style="font-size:12px;margin:14px 0 4px">Öğrenilen sinyal ağırlıkları (yeşil = yükselişe, kırmızı = düşüşe katkı):</div>${bars}`
    + `<div class="muted" style="font-size:12px;margin-top:12px;padding:10px 12px;border:1px solid var(--line);border-radius:12px;background:rgba(244,185,66,.06)">
        ⚠️ <b>Dürüst not:</b> Bu ≈3 aylık yön için <b>gerçek (out-of-sample)</b> bir sonuçtur ve trend-takip baseline'ını <b>${edge}</b> geçer. Piyasa gürültülü olduğundan hiçbir teknik model %90+ isabet <i>veremez</i> — öyle bir sayı görürseniz o model geleceğe bakıyordur (veri sızıntısı). Daha yükseği için temel analiz + piyasa rejimi verisi gerekir.
        Seçici moddaki yüksek isabetler (örn. %82) <b>düşük kapsamla</b> gelir (%2-3 gün) — az ama güçlü sinyal, sürekli işlem değildir.
      </div>`;
}

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
    renderChips();
    showTab('analiz');
    loadBrowseExchanges();
    await loadNewsSources();
    await loadUniverses();
    await loadSettings();
    await loadHistory();
    await loadDashboard();
    await loadModelInfo();
    // Restore the last analysis of this session (server keeps it in memory), so
    // reopening the tab doesn't lose the results.
    await refreshState();
}

loadChartPrefs();
initializeForSession();
</script></body></html>"""