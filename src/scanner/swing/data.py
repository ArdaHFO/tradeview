"""Daily-bar data for swing trading: free, keyless, and deep enough to prove things.

The intraday side of this repo is boxed in by its data: Polygon's free tier caps
at 5 requests/minute and serves no tick history, so order flow had to be
synthesised from bar shape and no setup ever reached a decisive sample. Daily
bars have none of those problems -- yfinance serves decades of history for
thousands of symbols with no API key -- and swing horizons do not need tick data
to begin with. That is the whole reason this module exists.

Two biases to keep in view, because they are what make daily backtests lie:

  * *Survivorship* -- the S&P 500 membership list is TODAY's. Backtesting it
    over past years quietly assumes you knew in advance which companies would
    still be in the index, which flatters returns. `universe()` reports this
    rather than hiding it; the honest mitigations are a broader universe or
    point-in-time constituent data (not free).
  * *Adjustment* -- prices are split- and dividend-adjusted (`auto_adjust`).
    That is correct for return calculations but means historical prices are not
    what actually printed on the tape.
"""
from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/daily")
CACHE_TTL_HOURS = 20            # a trading day's worth: re-pull once per session
WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
UA = {"User-Agent": "Mozilla/5.0 (trading-bot research script)"}

# Enough liquid large caps to run on when Wikipedia is unreachable. Not a real
# index -- just a fallback so the pipeline is never hard-blocked on a scrape.
FALLBACK_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "JPM", "V",
    "UNH", "XOM", "MA", "PG", "JNJ", "HD", "COST", "ABBV", "MRK", "WMT",
    "CVX", "PEP", "KO", "ADBE", "CRM", "BAC", "TMO", "MCD", "CSCO", "ACN",
    "LIN", "ABT", "PFE", "DHR", "INTC", "VZ", "TXN", "QCOM", "NKE", "AMD",
    "PM", "INTU", "UNP", "RTX", "LOW", "SPGI", "HON", "IBM", "GE", "CAT",
]


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.pkl"


def _fresh(path: Path, ttl_hours: float = CACHE_TTL_HOURS) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def universe(name: str = "sp500", use_cache: bool = True) -> list[str]:
    """Symbol list to trade. Returns TODAY's membership -- see survivorship note."""
    if name == "fallback":
        return list(FALLBACK_UNIVERSE)
    path = _cache_path(f"universe_{name}")
    if use_cache and _fresh(path, ttl_hours=24 * 7):
        return list(pd.read_pickle(path))
    try:
        resp = requests.get(WIKI_SP500, headers=UA, timeout=30)
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        # Wikipedia writes class shares as BRK.B; yfinance wants BRK-B.
        symbols = sorted({str(s).replace(".", "-").strip()
                          for s in table["Symbol"] if str(s).strip()})
        pd.to_pickle(symbols, path)
        return symbols
    except Exception as exc:                    # scrape is best-effort, never fatal
        log.warning("S&P 500 listesi alinamadi (%s) -- fallback evren kullaniliyor",
                    exc)
        if path.exists():
            return list(pd.read_pickle(path))
        return list(FALLBACK_UNIVERSE)


def load_daily(symbols: list[str], years: float = 10.0,
               use_cache: bool = True, batch: int = 100,
               progress=None) -> dict[str, pd.DataFrame]:
    """Download adjusted daily OHLCV, one tidy frame per symbol.

    Cached to disk by (symbol-set, span) so repeated backtests over the same
    window cost nothing. Symbols that return no data (delisted, renamed, bad
    ticker) are dropped with a log line rather than failing the run.
    """
    import yfinance as yf

    end = date.today()
    start = end - timedelta(days=int(years * 365.25) + 5)
    key = f"bars_{len(symbols)}_{years}_{hash(tuple(sorted(symbols))) & 0xFFFFFF:06x}"
    path = _cache_path(key)
    if use_cache and _fresh(path):
        if progress:
            progress(f"önbellekten yükleniyor ({path.name})")
        return pd.read_pickle(path)

    frames: dict[str, pd.DataFrame] = {}
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        if progress:
            progress(f"indiriliyor {i + 1}-{min(i + batch, len(symbols))} "
                     f"/ {len(symbols)}")
        raw = yf.download(chunk, start=start.isoformat(), end=end.isoformat(),
                          interval="1d", auto_adjust=True, progress=False,
                          threads=True, group_by="ticker")
        for sym in chunk:
            try:
                df = raw[sym] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if len(df) < 260:               # under a year of history: not usable
                continue
            frames[sym] = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    if not frames:
        raise RuntimeError("hiçbir sembol için veri indirilemedi (ağ sorunu?)")
    pd.to_pickle(frames, path)
    if progress:
        progress(f"{len(frames)} sembol hazır, önbelleğe yazıldı")
    return frames


def trading_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Union of all symbols' dates: the sessions the backtest steps through."""
    idx = pd.DatetimeIndex([])
    for df in frames.values():
        idx = idx.union(df.index)
    return idx.sort_values()
