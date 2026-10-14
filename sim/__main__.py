"""Run the baseline quoter across fill models and latencies for one day.

Usage:
  python -m sim --date 2026-10-14
  python -m sim --date 2026-10-14 --hours 2 --latency-ms 150,10,1 --venue binance --symbol BTCUSDT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .analysis import summarise
from .data import load_day
from .engine import MODELS, Engine
from .strategies import JoinQuoter

COLUMNS = [
    ("model", "{}"), ("latency_ms", "{:g}"), ("fills", "{}"), ("fill_share_of_orders", "{:.1%}"),
    ("median_wait_s", "{:.1f}"), ("spread_captured_bps", "{:+.2f}"), ("adverse_1s_bps", "{:+.2f}"),
    ("adverse_30s_bps", "{:+.2f}"), ("pnl_gross_bps", "{:+.2f}"), ("pnl_gross_usd", "{:+.2f}"),
]


def fmt(row: dict, key: str, f: str) -> str:
    v = row.get(key)
    if v is None:
        return "-"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--venue", default="binance")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--hours", type=float, default=None, help="only use the first N hours")
    ap.add_argument("--latency-ms", default="150,10,1")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--size", type=float, default=0.01)
    ap.add_argument("--max-pos", type=float, default=0.05)
    ap.add_argument("--queue-power", type=float, default=1.0)
    args = ap.parse_args()

    md = load_day(args.clean, args.venue, args.symbol, args.date)
    if args.hours:
        md = md.clip_hours(args.hours)
    print(f"{args.venue} {args.symbol} {args.date}: {len(md.g_start):,} book messages, {len(md.t_exch):,} trades")

    rows = []
    for lat in [float(x) for x in args.latency_ms.split(",")]:
        for model in args.models.split(","):
            t0 = time.time()
            res = Engine(md, JoinQuoter(args.size, args.max_pos), int(lat * 1e6), model, args.queue_power).run()
            row = summarise(res)
            rows.append(row)
            print(f"  {model:<12} {lat:>6g} ms  {row['fills']:>6} fills  ({time.time() - t0:.0f}s)")

    print()
    print("  ".join(f"{k:>14}" for k, _ in COLUMNS))
    for r in rows:
        print("  ".join(f"{fmt(r, k, f):>14}" for k, f in COLUMNS))

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"sim_{args.venue}_{args.symbol}_{args.date}{'_' + str(args.hours) + 'h' if args.hours else ''}.json"
    path.write_text(json.dumps({"args": vars(args), "results": rows}, indent=2))
    print(f"\nFull results (fees, markouts at all horizons, PnL split) saved to {path}")


if __name__ == "__main__":
    main()
