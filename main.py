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


if __name__ == "__main__":
    main()
