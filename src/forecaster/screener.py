"""Market screener: scan a preset universe for the strongest technical setups.

Deliberately technical-only (no Groq/news) so scanning 20+ symbols stays fast
and free — a screener is about surfacing setups to then analyze in depth, which
the full news+technical pipeline already does per symbol.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import Config
from .models import Direction
from .technical.data import ALLOWED_TIMEFRAMES, fetch_bars
from .technical.indicators import rsi
from .technical.scorer import score_technical

log = logging.getLogger(__name__)

# Curated, liquid, well-known names per market. Kept ~20/list so a synchronous
# scan finishes in a few seconds. (symbol, display name).
UNIVERSES: dict[str, dict] = {
    "bist": {
        "label": "BIST Popüler (Türkiye)",
        "symbols": [
            ("THYAO.IS", "Türk Hava Yolları"), ("ASELS.IS", "Aselsan"),
            ("GARAN.IS", "Garanti BBVA"), ("AKBNK.IS", "Akbank"),
            ("SISE.IS", "Şişecam"), ("KCHOL.IS", "Koç Holding"),
            ("EREGL.IS", "Ereğli Demir Çelik"), ("BIMAS.IS", "BİM"),
            ("TUPRS.IS", "Tüpraş"), ("SAHOL.IS", "Sabancı Holding"),
            ("FROTO.IS", "Ford Otosan"), ("PGSUS.IS", "Pegasus"),
            ("TCELL.IS", "Turkcell"), ("YKBNK.IS", "Yapı Kredi"),
            ("KOZAL.IS", "Koza Altın"), ("PETKM.IS", "Petkim"),
            ("TOASO.IS", "Tofaş"), ("ISCTR.IS", "İş Bankası"),
            ("ARCLK.IS", "Arçelik"), ("HEKTS.IS", "Hektaş"),
        ],
    },
    "us": {
        "label": "ABD Popüler",
        "symbols": [
            ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
            ("AMZN", "Amazon"), ("GOOGL", "Alphabet"), ("META", "Meta"),
            ("TSLA", "Tesla"), ("AMD", "AMD"), ("NFLX", "Netflix"),
            ("JPM", "JPMorgan"), ("V", "Visa"), ("DIS", "Disney"),
            ("BA", "Boeing"), ("KO", "Coca-Cola"), ("PFE", "Pfizer"),
            ("XOM", "ExxonMobil"), ("WMT", "Walmart"), ("INTC", "Intel"),
            ("CRM", "Salesforce"), ("PYPL", "PayPal"),
        ],
    },
    "eu": {
        "label": "Avrupa Popüler",
        "symbols": [
            ("SAP.DE", "SAP"), ("SIE.DE", "Siemens"), ("MC.PA", "LVMH"),
            ("OR.PA", "L'Oréal"), ("ASML.AS", "ASML"), ("AIR.PA", "Airbus"),
            ("BAS.DE", "BASF"), ("ALV.DE", "Allianz"), ("BMW.DE", "BMW"),
            ("TTE.PA", "TotalEnergies"), ("AZN.L", "AstraZeneca"),
            ("SHEL.L", "Shell"), ("VOW3.DE", "Volkswagen"), ("DTE.DE", "Deutsche Telekom"),
        ],
    },
}


def list_universes() -> list[dict]:
    return [{"id": key, "label": val["label"], "count": len(val["symbols"])}
            for key, val in UNIVERSES.items()]


def universe_symbols(key: str) -> list[dict]:
    """The (symbol, name) list for one universe, for browsing in the UI when a
    user can't recall a ticker."""
    entries = UNIVERSES.get(key, UNIVERSES["bist"])["symbols"]
    return [{"symbol": sym, "name": name} for sym, name in entries]


def _signal_label(score: float) -> str:
    if score >= 0.5:
        return "Güçlü Al"
    if score >= 0.15:
        return "Al"
    if score > -0.15:
        return "Nötr"
    if score > -0.5:
        return "Sat"
    return "Güçlü Sat"


def _direction(score: float) -> str:
    if score > 0.15:
        return Direction.UP.value
    if score < -0.15:
        return Direction.DOWN.value
    return Direction.NEUTRAL.value


def _scan_one(symbol: str, name: str, cfg: Config, timeframe: str, model=None,
              market_cache: dict | None = None) -> dict | None:
    try:
        bars = fetch_bars(symbol, cfg, timeframe)
    except Exception as exc:
        log.warning("screener: fetch failed for %s: %s", symbol, exc)
        return None
    if len(bars) < 50:
        return None
    verdict = score_technical(symbol, bars)
    closes = [b.close for b in bars]
    rsi_series = rsi(closes, 14)
    rsi_last = next((v for v in reversed(rsi_series) if v is not None), None)
    row = {
        "symbol": symbol,
        "name": name,
        "score": round(verdict.score, 3),
        "direction": _direction(verdict.score),
        "signal": _signal_label(verdict.score),
        "price": round(closes[-1], 2),
        "rsi": round(rsi_last, 1) if rsi_last is not None else None,
        "model_proba": None,
    }
    # The learned model's P(up over ~3mo), if a model is available. Screener has
    # no news, so the news feature is 0 — a technical-only estimate.
    if model is not None:
        from .learning.features import features_from_bars, market_index_for
        market_bars = (market_cache or {}).get(market_index_for(symbol))
        feat = features_from_bars(bars, market_bars=market_bars)
        if feat is not None:
            row["model_proba"] = round(model.predict_from_dict(feat) * 100, 1)
    return row


def scan_symbols(entries: list[tuple[str, str]], cfg: Config, timeframe: str = "1d") -> list[dict]:
    """Technically score an explicit (symbol, name) list, ranked by score desc."""
    if timeframe not in ALLOWED_TIMEFRAMES:
        timeframe = "1d"
    from .learning.features import market_index_for
    from .learning.train import load_model
    model = load_model(cfg.model_path)
    # One index fetch per market so every symbol gets regime features cheaply.
    market_cache: dict[str, list] = {}
    if model is not None:
        for index in {market_index_for(sym) for sym, _ in entries}:
            try:
                market_cache[index] = fetch_bars(index, cfg, timeframe)
            except Exception as exc:
                log.warning("screener: market index fetch failed for %s: %s", index, exc)
                market_cache[index] = []
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_scan_one, sym, name, cfg, timeframe, model, market_cache)
                   for sym, name in entries]
        for fut in as_completed(futures):
            row = fut.result()
            if row is not None:
                results.append(row)
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def scan(universe: str, cfg: Config, timeframe: str = "1d") -> list[dict]:
    """Score every symbol in a preset universe, ranked by score desc (strongest
    bullish first, strongest bearish last)."""
    entries = UNIVERSES.get(universe, UNIVERSES["bist"])["symbols"]
    return scan_symbols(entries, cfg, timeframe)
