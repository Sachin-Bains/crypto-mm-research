"""Run every interval method on the out-of-sample forecasts and compare them.

Usage:
  python -m forecast.intervals --dates 2026-10-14
  python -m forecast.intervals --dates 2026-10-14,2026-10-15 --model boosted --alpha 0.1

Reads data/results/forecast_oos_<first>_<last>.parquet written by forecast.evaluate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .conformal import METHODS, evaluate, intervals
from .features import HORIZONS_S

COLS = [("coverage", "{:.1%}"), ("width_bps", "{:.2f}"), ("interval_score", "{:.2f}"), ("cov_calm", "{:.1%}"),
        ("cov_stressed", "{:.1%}"), ("cov_top5pct_vol", "{:.1%}"), ("worst_10min_cov", "{:.1%}"),
        ("share_10min_below_80", "{:.1%}")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--model", default="ridge", choices=["ridge", "boosted"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--window-min", type=float, default=10)
    ap.add_argument("--gamma", type=float, default=0.005)
    args = ap.parse_args()

    days = [x.strip() for x in args.dates.split(",") if x.strip()]
    tag = f"{days[0]}_{days[-1]}"
    tbl = pq.read_table(Path("data/results") / f"forecast_oos_{tag}.parquet")
    d = {c: tbl.column(c).to_numpy() for c in tbl.column_names}
    t = d["recv_ns"]
    vol = np.maximum(d["rv_30s"].astype(np.float64), 0.1)   # floor: BTC can sit still for 30 s

    report, cols = {}, {"recv_ns": t}
    print(f"Target coverage {1 - args.alpha:.0%}, model: {args.model}, window {args.window_min:g} min")
    for h in HORIZONS_S:
        tg = f"y_{h}s"
        y = d[tg].astype(np.float64)
        pred = d[f"pred_{args.model}_{tg}"].astype(np.float64)
        t0 = time.time()
        hw = intervals(t, y, pred, vol, h * 10**9, alpha=args.alpha, window_ns=int(args.window_min * 60e9),
                       gamma=args.gamma)
        print(f"\n{tg}  ({time.time() - t0:.0f}s)")
        print(f"{'method':<10}" + "".join(f"{k:>22}" for k, _ in COLS))
        for m in METHODS:
            res = evaluate(t, y, pred, hw[m], vol, alpha=args.alpha)
            report[f"{tg}/{m}"] = res
            cols[f"hw_{m}_{tg}"] = hw[m].astype(np.float32)
            row = "".join(f"{(f.format(res[k]) if k in res and np.isfinite(res[k]) else '-'):>22}" for k, f in COLS)
            print(f"{m:<10}{row}")
        cols[tg] = y.astype(np.float32)
        cols[f"pred_{tg}"] = pred.astype(np.float32)

    out = Path("data/results")
    (out / f"intervals_{args.model}_{tag}.json").write_text(json.dumps(report, indent=2))
    pq.write_table(pa.table(cols), out / f"intervals_{args.model}_{tag}.parquet", compression="zstd")
    print(f"\nSaved to {out}/intervals_{args.model}_{tag}.*")


if __name__ == "__main__":
    main()
