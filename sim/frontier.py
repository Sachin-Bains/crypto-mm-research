"""Fill volume against adverse selection, for every method across its one dial.

Usage:
  python -m sim.frontier --date 2026-10-21 --intervals-dates 2026-10-14,2026-10-22

Each method trades off how much it trades against how adversely selected its
fills are. Comparing single points at one setting can flatter whichever method
happened to be set more cautiously, so this sweeps each method's dial and
compares the whole curves:

  random_off_pull   withdrawal probability        (no information)
  vol_pull          volatility threshold          (volatility timing only)
  conformal         adverse-selection budget B    (version 2: forecast + interval, one rule)
  join, full_pull   single points                 (baseline, version 1)

Everything else (latency 150 ms, queue fill model, size, limits, inventory rule)
is the same for all of them. Results go to data/results/frontier_<date>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .analysis import hourly_markout_usd, hourly_notional_usd, summarise
from .data import load_day
from .engine import QUEUE, Engine
from .signal import Signal
from .strategies import VARIANTS, ConformalBudgetQuoter, ForecastQuoter

GRID = {
    "random_off_pull": [0.2, 0.35, 0.5, 0.65, 0.8],
    "vol_pull": [0.8, 1.0, 1.2, 1.5, 2.0],
    "conformal": [0.1, 0.2, 0.3, 0.4],
}
KEEP = ("fills", "notional_usd", "markout_1s_bps", "adverse_1s_bps", "markout_5s_bps", "adverse_5s_bps",
        "markout_30s_bps", "pnl_gross_bps", "pnl_gross_usd", "pnl_net_usd [MM rebate (-0.5 bp)]",
        "pnl_net_usd [retail (2.0 bp)]", "spread_captured_usd", "final_position")


def load_signal(args):
    days = [d.strip() for d in args.intervals_dates.split(",") if d.strip()]
    tag = f"{days[0]}_{days[-1]}"
    iv = pq.read_table(Path("data/results") / f"intervals_{args.model}_{tag}.parquet")
    tg = f"y_{args.horizon}s"
    sig_t = iv.column("recv_ns").to_numpy()
    pred = iv.column(f"pred_{tg}").to_numpy().astype(np.float64)
    hw = iv.column(f"hw_{args.method}_{tg}").to_numpy().astype(np.float64)
    fo = pq.read_table(Path("data/results") / f"forecast_oos_{tag}.parquet", columns=["recv_ns", "rv_30s"])
    ft, fv = fo.column("recv_ns").to_numpy(), fo.column("rv_30s").to_numpy().astype(np.float64)
    idx = np.clip(np.searchsorted(ft, sig_t), 0, len(ft) - 1)
    vol = np.where(ft[idx] == sig_t, np.maximum(fv[idx], 0.1), np.nan)
    md = load_day(args.clean, "binance", "BTCUSDT", args.date)
    if not md.has_bbo:
        raise SystemExit("needs the real-time best bid/offer (bbo.parquet)")
    return md, Signal(sig_t, pred, hw, md.b_exch, md.b_recv, vol=vol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--intervals-dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--model", default="ridge")
    ap.add_argument("--method", default="adaptive")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--latency-ms", type=float, default=150)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--only", default=None, help="comma list of methods to run (default: all)")
    args = ap.parse_args()

    md, signal = load_signal(args)
    L = int(args.latency_ms * 1e6)
    base = dict(size=0.01, max_pos=0.05, tick=0.1, theta=0.25, depth_k=5.0, rho_max=2.5)
    plan = [("join", None), ("full_pull", None)]
    for m, dials in GRID.items():
        plan += [(m, d) for d in dials]
    if args.only:
        keep = set(args.only.split(","))
        plan = [p for p in plan if p[0] in keep]

    points = []
    print(f"binance BTCUSDT {args.date}, {len(plan)} runs at {args.latency_ms:g} ms")
    for method, dial in plan:
        if method == "conformal":
            strat = ConformalBudgetQuoter(signal, L, budget_bp=dial, gamma=args.gamma)
        else:
            params = dict(base)
            params.update(VARIANTS[method])
            if method == "random_off_pull":
                params["random_off"] = dial
            if method == "vol_pull":
                params["vol_max"] = dial
            strat = ForecastQuoter(signal, L, **params)
        t0 = time.time()
        res = Engine(md, strat, L, QUEUE).run()
        s = summarise(res)
        pt = {"method": method, "dial": dial, **{k: s.get(k) for k in KEEP},
              "quoting": strat.quoting / max(strat.calls, 1),
              "one_sided_share": strat.one_sided / max(strat.calls, 1),
              "hourly_markout5s_usd": {str(k): v for k, v in hourly_markout_usd(res).items()},
              "hourly_notional_usd": {str(k): v for k, v in hourly_notional_usd(res).items()}}
        if method == "conformal":
            pt["conformal"] = strat.report()
        points.append(pt)
        extra = ""
        if method == "conformal":
            r = pt["conformal"]
            extra = f"  budget {dial:.2f} -> realised {r['realised_loss_bp']:.3f} bp, lam {r['lam_final']:.2f}, at bound {r['share_updates_at_bound']:.0%}"
        print(f"  {method:<16} {str(dial):>6}  {s['fills']:>6} fills  markout 5s {s.get('markout_5s_bps', float('nan')):+.3f} bp"
              f"  ({time.time() - t0:.0f}s){extra}")

    out = Path("data/results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"frontier_{args.date}.json"
    path.write_text(json.dumps({"args": vars(args), "grid": GRID, "points": points}, indent=1))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
