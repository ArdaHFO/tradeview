"""Benchmark comparison: does the strategy beat simply owning the index?

A long-only strategy backtested across 2016-2026 is graded on a decade that was
mostly a bull market, so "it made money" and "it was worth running" are entirely
different claims. Positive expectancy per trade does not settle it either: the
first strategy in this repo to clear statistical significance still returned
+130% against SPY's +305% over the same window, i.e. it beat coin-flipping and
lost badly to doing nothing.

So every swing result is reported against buy-and-hold on the same dates, on
three axes: total return, drawdown, and return per unit of drawdown. A strategy
earns its complexity only by winning at least one of them convincingly.
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "SPY"


def _max_drawdown_pct(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float(((curve - peak) / peak * 100.0).min())


def _cagr(first: float, last: float, days: float) -> float:
    if first <= 0 or days <= 0:
        return 0.0
    return ((last / first) ** (365.25 / days) - 1.0) * 100.0


class Stats(dict):
    """Plain dict so callers can serialise it straight to JSON."""


def series_stats(curve: pd.Series) -> Stats:
    curve = curve.dropna()
    if len(curve) < 2:
        return Stats(total_pct=0.0, max_dd_pct=0.0, cagr_pct=0.0, calmar=0.0)
    first, last = float(curve.iloc[0]), float(curve.iloc[-1])
    days = (curve.index[-1] - curve.index[0]).days
    total = (last / first - 1.0) * 100.0
    dd = _max_drawdown_pct(curve)
    cagr = _cagr(first, last, days)
    return Stats(total_pct=total, max_dd_pct=dd, cagr_pct=cagr,
                 calmar=(cagr / abs(dd)) if dd else 0.0)


def load_benchmark(start, end, symbol: str = DEFAULT_BENCHMARK) -> pd.Series | None:
    """Adjusted closes for the benchmark over the backtest's own dates."""
    import yfinance as yf
    try:
        df = yf.download(symbol, start=pd.Timestamp(start).date().isoformat(),
                         end=(pd.Timestamp(end) + pd.Timedelta(days=1))
                         .date().isoformat(),
                         interval="1d", auto_adjust=True, progress=False)
    except Exception as exc:                    # offline: skip, do not fail the run
        log.warning("benchmark %s indirilemedi: %s", symbol, exc)
        return None
    if df is None or df.empty:
        return None
    return df["Close"].squeeze().dropna()


def compare(equity_curve: pd.Series,
            symbol: str = DEFAULT_BENCHMARK) -> dict | None:
    """Strategy vs buy-and-hold over identical dates. None if unavailable."""
    curve = equity_curve.dropna()
    if len(curve) < 2:
        return None
    bench = load_benchmark(curve.index[0], curve.index[-1], symbol)
    if bench is None or len(bench) < 2:
        return None
    strat = series_stats(curve)
    bh = series_stats(bench)
    return {"symbol": symbol, "strategy": strat, "benchmark": bh,
            "beats_return": strat["total_pct"] > bh["total_pct"],
            "beats_drawdown": abs(strat["max_dd_pct"]) < abs(bh["max_dd_pct"]),
            "beats_calmar": strat["calmar"] > bh["calmar"]}


def format_comparison(cmp: dict | None) -> str:
    if cmp is None:
        return "  (benchmark karşılaştırması yapılamadı — ağ yok?)"
    s, b, sym = cmp["strategy"], cmp["benchmark"], cmp["symbol"]
    mark = lambda ok: "✅" if ok else "❌"          # noqa: E731
    lines = [
        f"  --- {sym} al-tut karşılaştırması ---",
        f"  {'':18s}{'strateji':>12s}{sym:>12s}",
        f"  {'Toplam getiri':18s}{s['total_pct']:>11.1f}%{b['total_pct']:>11.1f}%"
        f"  {mark(cmp['beats_return'])}",
        f"  {'Max drawdown':18s}{s['max_dd_pct']:>11.1f}%{b['max_dd_pct']:>11.1f}%"
        f"  {mark(cmp['beats_drawdown'])}",
        f"  {'Yıllık (CAGR)':18s}{s['cagr_pct']:>11.1f}%{b['cagr_pct']:>11.1f}%",
        f"  {'Calmar (CAGR/DD)':18s}{s['calmar']:>12.2f}{b['calmar']:>12.2f}"
        f"  {mark(cmp['beats_calmar'])}",
    ]
    if not (cmp["beats_return"] or cmp["beats_calmar"]):
        lines.append(f"  ⛔ Strateji {sym} al-tut'u hiçbir eksende geçemiyor — "
                     "bu karmaşıklığı taşımaya değmez.")
    elif not cmp["beats_return"]:
        lines.append(f"  ⚠ Mutlak getiri {sym}'nin altında; sadece risk-ayarlı "
                     "tarafta öne geçiyor.")
    return "\n".join(lines)
