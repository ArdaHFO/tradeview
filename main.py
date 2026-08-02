"""CLI entry point.

Usage:
  python main.py demo            # synthetic end-to-end run, no API key needed
  python main.py screen          # stage-1 watchlist from Polygon (needs key)
  python main.py live            # stage-1 + live stage-2 stream (needs RT plan)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")
if hasattr(sys.stdout, "reconfigure"):       # Windows cp1252 console: allow emoji
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scanner.alerts.telegram import TelegramAlerter, format_signal
from scanner.config import load_config
from scanner.session import session_elapsed_fraction
from scanner.stage1_screener import screen


def cmd_demo(cfg) -> int:
    from scanner.demo import run_demo
    watchlist, signals = run_demo(cfg)
    print("=== STAGE 1 — Watchlist ===")
    for w in watchlist:
        flt = f"{w.float_shares/1e6:.0f}M" if w.float_shares else "?"
        print(f"  {w.symbol:6s} rvol {w.rvol:4.1f}  gap {w.gap_pct:+6.1f}%  "
              f"${w.price:<7.2f} float {flt:>5s}  heat {w.score:.0f}")
    print(f"\n=== STAGE 2 — Signals ({len(signals)}) ===")
    for s in signals:
        print()
        print(format_signal(s))
    if not signals:
        print("  (no signals emitted — check thresholds)")
        return 1
    return 0


def cmd_screen(cfg) -> int:
    from scanner.data.provider import PolygonProvider
    provider = PolygonProvider(cfg.polygon_api_key)
    snapshots = provider.full_market_snapshot()
    elapsed = session_elapsed_fraction(datetime.now(timezone.utc))
    # First pass on cheap data, then enrich only the survivors and re-screen.
    rough = screen(snapshots, cfg.screener, elapsed)
    shortlist = [s for s in snapshots if s.symbol in {w.symbol for w in rough}]
    provider.enrich_shortlist(shortlist)
    watchlist = screen(shortlist, cfg.screener, elapsed)
    print(f"Watchlist ({len(watchlist)}):")
    for w in watchlist:
        flt = f"{w.float_shares/1e6:.0f}M" if w.float_shares else "?"
        print(f"  {w.symbol:6s} rvol {w.rvol:4.1f}  gap {w.gap_pct:+6.1f}%  "
              f"${w.price:<7.2f} float {flt:>5s}  heat {w.score:.0f}")
    return 0


def cmd_live(cfg) -> int:
    from scanner.data.provider import PolygonProvider
    from scanner.engine import SignalEngine
    from scanner.signal.recorder import SignalRecorder

    provider = PolygonProvider(cfg.polygon_api_key)
    snapshots = provider.full_market_snapshot()
    elapsed = session_elapsed_fraction(datetime.now(timezone.utc))
    rough = screen(snapshots, cfg.screener, elapsed)
    shortlist = [s for s in snapshots if s.symbol in {w.symbol for w in rough}]
    provider.enrich_shortlist(shortlist)
    watchlist = screen(shortlist, cfg.screener, elapsed)
    if not watchlist:
        print("Nothing in play right now.")
        return 0
    symbols = [w.symbol for w in watchlist]
    print(f"Streaming {len(symbols)} symbols: {', '.join(symbols)}")

    alerter = TelegramAlerter(cfg.telegram_bot_token, cfg.telegram_chat_id)
    recorder = SignalRecorder(cfg.db_path)
    engine = SignalEngine(cfg, watchlist, alerter=alerter, recorder=recorder)
    provider.stream_watchlist(symbols, engine.on_trade, engine.on_quote,
                              engine.on_bar)
    return 0


def cmd_dashboard(cfg) -> int:
    import uvicorn

    from scanner.dashboard import create_app
    app = create_app(cfg)
    print("Dashboard: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    return 0


def cmd_backtest(cfg, day_str: str | None, top_n: int, n_days: int,
                 save_trades: str | None = None) -> int:
    from datetime import date, timedelta

    from scanner.backtest import (Backtester, format_multi_report,
                                  format_report)
    if day_str:
        trading_day = date.fromisoformat(day_str)
    else:
        trading_day = datetime.now(timezone.utc).date() - timedelta(days=1)
        while trading_day.weekday() >= 5:
            trading_day -= timedelta(days=1)
    bt = Backtester(cfg)
    if n_days > 1:
        reports = bt.run_multi(trading_day, n_days, top_n=top_n,
                               progress=lambda m: print(f"  … {m}"))
        print()
        for rep in reports:
            print(format_report(rep))
            print()
        print(format_multi_report(reports))
    else:
        reports = [bt.run(trading_day, top_n=top_n,
                          progress=lambda m: print(f"  … {m}"))]
        print()
        print(format_report(reports[0]))
    if save_trades:
        from scanner.validation import save_trades as dump, trades_from_reports
        trades = trades_from_reports(reports)
        dump(trades, save_trades)
        print(f"\n{len(trades)} işlem kaydedildi → {save_trades}"
              f"  (doğrulama: python main.py validate --trades {save_trades})")
    return 0


def cmd_validate(cfg, log_path: str | None, trades_path: str | None,
                 iterations: int, setup_filter: str) -> int:
    """Bootstrap + Monte Carlo on a finished backtest: is the edge real?"""
    from scanner.validation import (load_trades, save_trades, trades_from_log,
                                    validate)
    if trades_path:
        trades = load_trades(trades_path)
        source = trades_path
    elif log_path:
        trades = trades_from_log(log_path)
        source = log_path
    else:
        print("hata: --log veya --trades ver (örn. --log backtest_baseline_30d.log)")
        return 2
    if setup_filter:
        keep = {s.strip().upper() for s in setup_filter.split(",")}
        trades = [t for t in trades if t.setup in keep]
    if len(trades) < 2:
        print(f"hata: {source} içinde doğrulanacak yeterli işlem yok "
              f"({len(trades)} bulundu)")
        return 1
    print(f"Kaynak: {source}\n")
    _, _, vd, report = validate(trades, cfg.risk.equity, iterations=iterations)
    print(report)
    if log_path and not trades_path:      # cache the parsed trades for reuse
        out = log_path.rsplit(".", 1)[0] + "_trades.json"
        save_trades(trades, out)
        print(f"\n  (ayrıştırılan işlemler kaydedildi → {out})")
    return 0 if vd.passed else 1


def cmd_swing(cfg, strategy_name: str, years: float, universe_name: str,
              max_positions: int, save_trades: str | None,
              iterations: int, oos_from: str = "") -> int:
    """Daily-bar swing backtest + the same statistical validation as intraday."""
    from scanner.swing import backtest as swing_bt
    from scanner.swing.data import load_daily, universe
    from scanner.swing.strategies import REGISTRY, build
    from scanner.validation import save_trades as dump, validate

    if strategy_name == "all":
        names = sorted(REGISTRY)
    else:
        names = [s.strip() for s in strategy_name.split(",")]
    say = lambda m: print(f"  … {m}")                       # noqa: E731

    symbols = universe(universe_name)
    print(f"Evren: {universe_name} — {len(symbols)} sembol, {years:g} yıl\n")
    frames = load_daily(symbols, years=years, progress=say)

    cfg_swing = swing_bt.SwingConfig(equity=cfg.risk.equity,
                                     max_positions=max_positions)
    failures = 0
    for name in names:
        try:
            strat = build(name)
        except KeyError as exc:
            print(f"hata: {exc}")
            return 2
        print()
        rep = swing_bt.run(frames, strat, cfg_swing, progress=say)
        print()
        print(swing_bt.format_report(rep, cfg_swing))
        if len(rep.trades) < 2:
            print("  (doğrulama için yeterli işlem yok)")
            failures += 1
            continue
        print()
        _, _, vd, text = validate(rep.trades, cfg_swing.equity,
                                  iterations=iterations)
        print(text)
        if oos_from:
            # Split by date and validate each half. The strategies here use
            # published parameters rather than ones fitted to this sample, so
            # this is checking selection bias (three candidates were tried and
            # the best kept), not classic parameter overfitting.
            ins = [t for t in rep.trades if t.day < oos_from]
            oos = [t for t in rep.trades if t.day >= oos_from]
            print(f"\n  ═══ ÖRNEKLEM DIŞI KONTROL (kesim {oos_from}) ═══")
            for label, subset in (("IN-SAMPLE ", ins), ("OUT-SAMPLE", oos)):
                if len(subset) < 2:
                    print(f"  {label}: yetersiz işlem ({len(subset)})")
                    continue
                b, _, v2, _ = validate(subset, cfg_swing.equity,
                                       iterations=iterations)
                mark = "✅" if v2.passed else "⛔"
                print(f"  {label}: {len(subset):5d} işlem | "
                      f"beklenti {b.expectancy_r.point:+.3f} R "
                      f"[{b.expectancy_r.lo:+.3f}, {b.expectancy_r.hi:+.3f}] | "
                      f"PF {b.profit_factor.point:.2f} {mark}")
            if len(oos) >= 2:
                _, _, v_oos, _ = validate(oos, cfg_swing.equity,
                                          iterations=iterations)
                if not v_oos.passed:
                    print("  ⛔ Örneklem dışında edge kayboluyor — in-sample "
                          "sonuç seçim yanlılığı olabilir.")
                    failures += 1
        if save_trades:
            path = (save_trades if len(names) == 1
                    else save_trades.rsplit(".", 1)[0] + f"_{name}.json")
            dump(rep.trades, path)
            print(f"\n  işlemler kaydedildi → {path}")
        failures += 0 if vd.passed else 1
    return 0 if failures == 0 else 1


def cmd_swing_scan(cfg, strategy_name: str, universe_name: str,
                   max_positions: int) -> int:
    """Today's swing entries from the latest closed daily bar."""
    from scanner.swing.backtest import SwingConfig
    from scanner.swing.data import load_daily, universe
    from scanner.swing.scan import format_scan, scan_today
    from scanner.swing.strategies import build

    if strategy_name == "all":
        strategy_name = "meanrev"          # the only one that cleared validation
    say = lambda m: print(f"  … {m}")                       # noqa: E731
    symbols = universe(universe_name)
    print(f"Evren: {universe_name} — {len(symbols)} sembol\n")
    # Two years is plenty of warmup for a 200-day mean and keeps the pull light.
    frames = load_daily(symbols, years=2, progress=say)
    cfg_swing = SwingConfig(equity=cfg.risk.equity, max_positions=max_positions)
    print()
    for name in [s.strip() for s in strategy_name.split(",")]:
        cands, bar_date = scan_today(frames, build(name), cfg_swing)
        print(format_scan(cands, bar_date, name, cfg_swing))
        print()
    return 0


def cmd_swing_positions(universe_name: str, add: str | None,
                        shares: int, price: float, stop: float,
                        strategy_name: str, remove: str | None) -> int:
    """Track open swing positions and say SELL or HOLD on each."""
    from scanner.swing.data import load_daily, universe
    from scanner.swing.positions import (Position, format_review,
                                         load_positions, review,
                                         save_positions)
    from scanner.swing.strategies import build

    positions = load_positions()

    if remove:
        keep = [p for p in positions if p.symbol.upper() != remove.upper()]
        if len(keep) == len(positions):
            print(f"{remove}: kayıtlı pozisyon bulunamadı")
            return 1
        save_positions(keep)
        print(f"{remove.upper()} kaldırıldı ({len(keep)} pozisyon kaldı)")
        return 0

    if add:
        sym = add.upper()
        if price <= 0 or shares <= 0:
            print("hata: --price ve --shares zorunlu (örn. --add AAPL "
                  "--shares 3 --price 308.91)")
            return 2
        if stop <= 0:
            # Reconstruct the strategy's own stop rather than inventing one.
            frames = load_daily(universe(universe_name), years=2,
                                progress=lambda m: print(f"  … {m}"))
            df = frames.get(sym)
            if df is None:
                print(f"hata: {sym} için veri yok (evren: {universe_name})")
                return 1
            from scanner.swing.strategies import add_indicators
            atr_val = float(add_indicators(df).iloc[-1]["atr14"])
            stop = price - build(strategy_name).stop_atr * atr_val
            print(f"  stop hesaplandı: {stop:.2f} "
                  f"({build(strategy_name).stop_atr:g} ATR)")
        positions = [p for p in positions if p.symbol != sym]
        positions.append(Position(symbol=sym, strategy=strategy_name,
                                  entry_date=datetime.now(timezone.utc)
                                  .date().isoformat(),
                                  entry_price=price, shares=shares,
                                  stop=round(stop, 2)))
        save_positions(positions)
        print(f"{sym} eklendi: {shares} adet @ {price:.2f}, stop {stop:.2f}")
        return 0

    if not positions:
        print(format_review([]))
        return 0
    symbols = sorted({p.symbol for p in positions})
    frames = load_daily(symbols, years=2, progress=lambda m: print(f"  … {m}"))
    print()
    print(format_review(review(positions, frames, build)))
    return 0


def cmd_alpaca_check(cfg) -> int:
    """Report what this Alpaca key can actually read — iex only, or full SIP."""
    from scanner.data.alpaca import AlpacaData, AlpacaError
    if not cfg.alpaca_key_id or not cfg.alpaca_secret_key:
        print("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY tanımlı değil.\n"
              "alpaca.markets üzerinden ücretsiz hesap açıp .env dosyasına ekle:\n"
              "  ALPACA_API_KEY_ID=...\n  ALPACA_API_SECRET_KEY=...")
        return 2
    try:
        api = AlpacaData(cfg.alpaca_key_id, cfg.alpaca_secret_key)
    except AlpacaError as exc:
        print(f"hata: {exc}")
        return 2
    print("Alpaca geçmiş veri yetkisi (tick verisi kritik — CVD'yi gerçek yapan bu):")
    result = api.probe_feeds()
    for feed, status in result.items():
        print(f"  {feed:4s} : {status}")
    if result.get("sip", "").startswith("OK"):
        print("\n  → SIP açık: backtest tam tape ile çalışabilir. "
              "ALPACA_FEED=sip yap.")
    elif result.get("iex", "").startswith("OK"):
        print("\n  → Sadece IEX: gerçek printler ama konsolide hacmin ~%2.5'i. "
              "CVD yönü anlamlı, mutlak hacim değil.")
    else:
        print("\n  → Hiçbir feed okunamadı; anahtarları kontrol et.")
    return 0


def cmd_today(cfg, top_n: int) -> int:
    from scanner.today import replay_today
    watchlist, results = replay_today(cfg, top_n=top_n)
    print(f"=== BUGÜNÜN WATCHLIST'İ ({len(watchlist)}) — Yahoo, anahtarsız ===")
    for w in watchlist:
        print(f"  {w.symbol:6s} rvol {w.rvol:5.1f}x  chg {w.gap_pct:+6.1f}%  "
              f"${w.price:<8.2f} heat {w.score:.0f}")
    print(f"\n=== BUGÜN ATEŞLENEN SİNYALLER ({len(results)}) ===")
    for r in results:
        s = r.signal
        print(f"  {s.ts.strftime('%H:%M')} {s.symbol:6s} {s.setup.value:14s} "
              f"{s.side.value:5s} entry {s.entry:8.2f} stop {s.stop:8.2f} "
              f"target {s.target:8.2f} → {r.exit_reason:6s} R {r.r_multiple:+5.2f} "
              f"PnL ${r.pnl:+8.2f}")
    if results:
        total = sum(r.pnl for r in results)
        wins = sum(1 for r in results if r.pnl > 0)
        print(f"\n  Bugün simülasyon: {len(results)} işlem | win "
              f"{wins/len(results)*100:.0f}% | PnL ${total:+,.2f}")
    else:
        print("  (sinyal yok)")
    return 0


def cmd_live_yf(cfg, top_n: int) -> int:
    from scanner.today import run_live_yahoo
    run_live_yahoo(cfg, top_n=top_n)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="US day-trading scanner")
    parser.add_argument("mode", choices=["demo", "screen", "live", "dashboard",
                                         "backtest", "validate", "today",
                                         "live-yf", "alpaca-check", "swing",
                                         "swing-scan", "swing-positions"])
    parser.add_argument("--date", help="backtest trading day (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=10,
                        help="backtest top-N watchlist symbols")
    parser.add_argument("--days", type=int, default=1,
                        help="backtest N trading days ending at --date")
    parser.add_argument("--disable", default="",
                        help="comma-separated setups to turn off (e.g. ORB)")
    parser.add_argument("--save-trades", metavar="PATH",
                        help="backtest: write trades to JSON for later validation")
    parser.add_argument("--log", help="validate: backtest console log to parse")
    parser.add_argument("--trades", help="validate: trades JSON to load")
    parser.add_argument("--iterations", type=int, default=10_000,
                        help="validate: bootstrap / monte carlo resamples")
    parser.add_argument("--setup", default="",
                        help="validate: only these setups (comma-separated)")
    parser.add_argument("--strategy", default="all",
                        help="swing: meanrev | breakout | trend | all")
    parser.add_argument("--years", type=float, default=10.0,
                        help="swing: years of daily history")
    parser.add_argument("--universe", default="sp500",
                        help="swing: sp500 | fallback")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="swing: concurrent position slots")
    parser.add_argument("--add", metavar="SYMBOL",
                        help="swing-positions: record a fill")
    parser.add_argument("--remove", metavar="SYMBOL",
                        help="swing-positions: drop a closed position")
    parser.add_argument("--shares", type=int, default=0,
                        help="swing-positions: share count for --add")
    parser.add_argument("--price", type=float, default=0.0,
                        help="swing-positions: fill price for --add")
    parser.add_argument("--stop", type=float, default=0.0,
                        help="swing-positions: stop for --add "
                             "(default: strategy's own ATR stop)")
    parser.add_argument("--oos-from", default="",
                        help="swing: split date (YYYY-MM-DD) for an "
                             "out-of-sample check")
    args = parser.parse_args()
    cfg = load_config()
    if args.disable:
        cfg.signal.disabled_setups = args.disable
    if args.mode == "backtest":
        return cmd_backtest(cfg, args.date, args.top, args.days,
                            args.save_trades)
    if args.mode == "validate":
        return cmd_validate(cfg, args.log, args.trades, args.iterations,
                            args.setup)
    if args.mode == "swing":
        return cmd_swing(cfg, args.strategy, args.years, args.universe,
                         args.max_positions, args.save_trades, args.iterations,
                         args.oos_from)
    if args.mode == "swing-positions":
        strat = "meanrev" if args.strategy == "all" else args.strategy
        return cmd_swing_positions(args.universe, args.add,
                                   args.shares, args.price, args.stop,
                                   strat, args.remove)
    if args.mode == "swing-scan":
        return cmd_swing_scan(cfg, args.strategy, args.universe,
                              args.max_positions)
    if args.mode == "alpaca-check":
        return cmd_alpaca_check(cfg)
    if args.mode == "today":
        return cmd_today(cfg, args.top)
    if args.mode == "live-yf":
        return cmd_live_yf(cfg, args.top)
    return {"demo": cmd_demo, "screen": cmd_screen, "live": cmd_live,
            "dashboard": cmd_dashboard}[args.mode](cfg)


if __name__ == "__main__":
    raise SystemExit(main())
