"""CLI: run | backfill | report | serve"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from forecaster.config import load_config  # noqa: E402
from forecaster.storage import backfill as backfill_mod  # noqa: E402
from forecaster.storage.recorder import PredictionRecorder  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="News + technical analysis stock direction forecaster")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="run the daily pipeline once")
    sub.add_parser("backfill", help="resolve prior predictions against latest prices")
    report_parser = sub.add_parser("report", help="print recent hit rate")
    report_parser.add_argument("--days", type=int, default=7)
    sub.add_parser("serve", help="run the web UI (FastAPI + uvicorn)")

    train_parser = sub.add_parser("train", help="train the learned fusion model on backtested history")
    train_parser.add_argument("--universe", default="us",
                              help="screener universe to train on: bist | us | eu | all (default us)")
    train_parser.add_argument("--timeframe", default="1d")
    train_parser.add_argument("--test-frac", type=float, default=0.3)

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "run":
        from forecaster.pipeline import run_daily
        predictions = run_daily(cfg)
        print(f"{len(predictions)} predictions generated.")
    elif args.command == "backfill":
        backfill_mod.run(cfg)
    elif args.command == "report":
        recorder = PredictionRecorder(cfg.db_path)
        hits, total = recorder.hit_rate(args.days)
        recorder.close()
        if total == 0:
            print(f"No resolved predictions in the last {args.days} days.")
        else:
            print(f"Last {args.days} days: {hits}/{total} hits ({hits/total:.0%})")
    elif args.command == "serve":
        import uvicorn
        from forecaster.webapp import create_app
        if not cfg.registration_code:
            logging.warning("REGISTRATION_CODE not set — anyone can register an account!")
        port = int(os.environ.get("PORT", "8000"))
        uvicorn.run(create_app(cfg), host="0.0.0.0", port=port)
    elif args.command == "train":
        import json
        from forecaster.screener import UNIVERSES
        from forecaster.learning.train import train_and_evaluate
        keys = list(UNIVERSES) if args.universe == "all" else [args.universe]
        symbols = [sym for k in keys for sym, _ in UNIVERSES.get(k, {}).get("symbols", [])]
        if not symbols:
            print(f"Unknown universe '{args.universe}'. Use: bist | us | eu | all")
            return
        print(f"Training on {len(symbols)} symbols from '{args.universe}' (~5y history each)...")
        _, report = train_and_evaluate(symbols, cfg, timeframe=args.timeframe,
                                        test_frac=args.test_frac, save_path=cfg.model_path)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
