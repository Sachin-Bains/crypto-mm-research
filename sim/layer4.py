"""Compare the Layer 4 strategies on one day, across latencies.

Usage:
  python -m sim.layer4 --date 2026-10-14
  python -m sim.layer4 --date 2026-10-14 --horizon 5 --method adaptive --model ridge --latency-ms 150,10,1

Needs the rebuilt day (python -m rebuild) and the interval file from
python -m forecast.intervals for the same dates (default: just --date).
All strategies quote only while a forecast exists, so they are compared over
the same hours. The fill model is the queue model unless --fill-model says otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .analysis import FEE_TIERS_BPS, summarise
from .data import load_day
from .engine import QUEUE, Engine
from .signal import Signal
from .strategies import VARIANTS, ForecastQuoter

COLUMNS = [("strategy", "{}"), ("latency_ms", "{:g}"), ("signal", "{:.0%}"), ("quoting", "{:.0%}"), ("fills", "{}"),
           ("adverse_1s_bps", "{:+.2f}"), ("adverse_5s_bps", "{:+.2f}"), ("markout_5s_bps", "{:+.2f}"),
           ("pnl_gross_bps", "{:+.2f}"), ("pnl_gross_usd", "{:+.2f}"), ("rebate_net_usd", "{:+.2f}")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--intervals-dates", default=None, help="the --dates used for forecast.intervals (default: --date)")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--model", default="ridge", choices=["ridge", "boosted"])
    ap.add_argument("--method", default="adaptive", help="interval method: static, rolling, scaled, regime, adaptive")
    ap.add_argument("--horizon", type=int, default=5, choices=[1, 5, 30])
    ap.add_argument("--latency-ms", default="150,10,1")
    ap.add_argument("--fill-model", default=QUEUE)
    ap.add_argument("--strategies", default=",".join(VARIANTS))
    ap.add_argument("--size", type=float, default=0.01)
    ap.add_argument("--max-pos", type=float, default=0.05)
    ap.add_argument("--tick", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=0.25)
    ap.add_argument("--depth-k", type=float, default=5.0)
    ap.add_argument("--rho-max", type=float, default=2.5)
    args = ap.parse_args()

    days = [d.strip() for d in (args.intervals_dates or args.date).split(",") if d.strip()]
    tag = f"{days[0]}_{days[-1]}"
    iv = pq.read_table(Path("data/results") / f"intervals_{args.model}_{tag}.parquet")
    tg = f"y_{args.horizon}s"
    sig_t = iv.column("recv_ns").to_numpy()
    pred = iv.column(f"pred_{tg}").to_numpy().astype(np.float64)
    hw = iv.column(f"hw_{args.method}_{tg}").to_numpy().astype(np.float64)

    md = load_day(args.clean, "binance", "BTCUSDT", args.date)
    if not md.has_bbo:
        raise SystemExit("Layer 4 needs the real-time best bid/offer (bbo.parquet)")
    signal = Signal(sig_t, pred, hw, md.b_exch, md.b_recv)
    win = signal.window()
    if win is None:
        raise SystemExit("No forecasts with intervals in that file")
    day_lo, day_hi = int(md.b_recv.min()), int(md.b_recv.max())
    in_day = (signal.t >= day_lo) & (signal.t <= day_hi)
    usable = in_day & np.isfinite(signal.f) & np.isfinite(signal.w)
    hours = usable.sum() * 0.1 / 3600
    print(f"binance BTCUSDT {args.date}: forecasts {args.model} {tg}, intervals {args.method}; "
          f"usable forecasts on this day: {hours:.1f} h of {in_day.sum() * 0.1 / 3600:.1f} h")
    if hours < 0.5:
        raise SystemExit("Fewer than 30 minutes of usable forecasts on this day: check the intervals file "
                         "(dates, method) before running the simulator.")

    rows = []
    for lat in [float(x) for x in args.latency_ms.split(",")]:
        L = int(lat * 1e6)
        for name in args.strategies.split(","):
            params = dict(size=args.size, max_pos=args.max_pos, tick=args.tick, theta=args.theta,
                          depth_k=args.depth_k, rho_max=args.rho_max)
            params.update(VARIANTS[name])
            strat = ForecastQuoter(signal, L, **params)
            t0 = time.time()
            res = Engine(md, strat, L, args.fill_model).run()
            row = summarise(res)
            row.update(strategy=name, signal=1 - strat.no_signal / max(strat.calls, 1),
                       quoting=strat.quoting / max(strat.calls, 1),
                       withdrawn_share=strat.withdrawn / max(strat.calls, 1),
                       one_sided_share=strat.one_sided / max(strat.calls, 1),
                       rebate_net_usd=row.get("pnl_net_usd [MM rebate (-0.5 bp)]"))
            rows.append(row)
            print(f"  {name:<9} {lat:>6g} ms  {row['fills']:>6} fills  ({time.time() - t0:.0f}s)")

    print()
    print("".join(f"{k:>16}" for k, _ in COLUMNS))
    for r in rows:
        cells = []
        for k, f in COLUMNS:
            v = r.get(k)
            try:
                cells.append(f.format(v) if v is not None else "-")
            except (TypeError, ValueError):
                cells.append(str(v))
        print("".join(f"{c:>16}" for c in cells))
    print("\nsignal: share of best-price changes at which a usable forecast existed")
    print("quoting: share of best-price changes at which the strategy had at least one quote out")
    print("rebate_net_usd: gross PnL plus a 0.5 bp maker rebate; other fee tiers are in the saved file")

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"layer4_{args.date}_{args.model}_{args.method}_{tg}.json"
    path.write_text(json.dumps({"args": vars(args), "fee_tiers_bps": FEE_TIERS_BPS, "results": rows}, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
