"""Where in the queue a market maker's fills come from, and what each kind costs.

Usage:
  python -m research.queue --dates 2026-10-15,2026-10-16,2026-10-17,2026-10-18 --intervals-dates 2026-10-14,2026-10-18

On a one-tick market the price you quote is fixed by the best bid or ask; what you control is
when you join the queue and whether you stay. This runs two strategies through the simulator
(join_pull: always at the best price; full_pull: version 1) and splits every fill by

  how         queue    our turn came: trades at our price used up the volume ahead of us
              through  a trade printed beyond our price (the level was swept)
              cross    the best price on the other side moved onto ours
  ahead       visible volume ahead of us when we joined the queue (BTC), in quintiles
  wait        time from joining to the fill
  liquidation fills in the 10 s after a liquidation episode starts, split by whether the
              forced flow was trading against us (it sold into our bid, or bought our ask)

Markout = side * (mid at t + h - fill price) / price, in bp, notional-weighted. Negative = the fill
lost money h seconds later. Writes data/results/research_queue_<first>_<last>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq

from sim.analysis import mid_at
from sim.engine import QUEUE, Engine
from sim.frontier import load_signal
from sim.strategies import VARIANTS, ForecastQuoter

from .liquidations import episodes

S = 1_000_000_000
HS = (1, 5, 30)


def fill_table(res) -> dict:
    f = res.fills
    t = np.array([x.t_ns for x in f], dtype=np.int64)
    out = {"t": t, "side": np.array([x.side for x in f], float), "price": np.array([x.price for x in f]),
           "qty": np.array([x.qty for x in f]), "ahead": np.array([x.ahead0 for x in f]),
           "wait_s": np.array([x.wait_ns for x in f]) / 1e9, "how": np.array([x.how for x in f])}
    for h in HS:
        m = mid_at(res, t + h * S)
        out[f"m{h}"] = out["side"] * (m - out["price"]) / out["price"] * 1e4
    return out


def stats(T: dict, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    w = T["qty"][mask] * T["price"][mask]
    r = {"n": n, "notional_usd": float(w.sum())}
    for h in HS:
        v = T[f"m{h}"][mask]
        ok = np.isfinite(v)
        r[f"markout_{h}s_bps"] = float(np.sum(v[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else float("nan")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--intervals-dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--latency-ms", type=float, default=150)
    ap.add_argument("--strategies", default="join_pull,full_pull")
    args = ap.parse_args()
    days = [d.strip() for d in args.dates.split(",") if d.strip()]
    L = int(args.latency_ms * 1e6)
    base = dict(size=0.01, max_pos=0.05, tick=0.1, theta=0.25, depth_k=5.0, rho_max=2.5)
    names = args.strategies.split(",")
    acc = {n: [] for n in names}
    eps_all = []
    for day in days:
        ns = SimpleNamespace(date=day, intervals_dates=args.intervals_dates, clean=args.clean, model="ridge",
                             method="adaptive", horizon=5)
        md, signal = load_signal(ns)
        ev = Path(args.clean) / "events" / day / "liquidations.parquet"
        if ev.exists():
            tab = pq.read_table(ev)
            c = {k: tab.column(k).to_numpy(zero_copy_only=False) for k in tab.column_names}
            k = (c["symbol"] == "BTC") & (c["exch_ns"] > 0) & (c["side"] != 0) & (c["qty"] > 0) & (c["price"] > 0)
            if k.any():
                eps_all += episodes({"t": c["exch_ns"][k].astype(np.int64), "side": c["side"][k].astype(int),
                                     "usd": c["qty"][k] * c["price"][k], "venue": c["venue"][k]})
        for name in names:
            p = dict(base)
            p.update(VARIANTS[name])
            res = Engine(md, ForecastQuoter(signal, L, **p), L, QUEUE).run()
            acc[name].append(fill_table(res))
            print(f"{day} {name:<10} {len(res.fills):>6} fills")

    ep_t = np.array([e["t0"] for e in eps_all], dtype=np.int64)
    ep_d = np.array([e["dir"] for e in eps_all], dtype=float)
    o = np.argsort(ep_t)
    ep_t, ep_d = ep_t[o], ep_d[o]
    report = {"dates": days, "latency_ms": args.latency_ms, "strategies": {}}
    for name, parts in acc.items():
        T = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        all_ = np.ones(len(T["t"]), bool)
        r = {"all": stats(T, all_), "how": {}, "ahead": [], "wait": [], "liquidation": {}}
        for how in ("queue", "through", "cross"):
            r["how"][how] = stats(T, T["how"] == how)
            r["how"][how]["share"] = float((T["how"] == how).mean())
        # quintiles by rank, so ties (many fills join an empty level) do not collapse the groups
        for grp in np.array_split(np.argsort(T["ahead"], kind="stable"), 5):
            m = np.zeros(len(T["t"]), bool)
            m[grp] = True
            a, b = (float(T["ahead"][grp].min()), float(T["ahead"][grp].max())) if len(grp) else (0.0, 0.0)
            r["ahead"].append({"range_btc": [a, b], **stats(T, m),
                               "share_queue_fills": float((T["how"][m] == "queue").mean()) if m.any() else None})
        for a, b in ((0, 0.25), (0.25, 1), (1, 5), (5, 30), (30, 1e9)):
            r["wait"].append({"range_s": [a, b], **stats(T, (T["wait_s"] >= a) & (T["wait_s"] < b))})
        if len(ep_t):
            i = np.searchsorted(ep_t, T["t"], side="right") - 1
            inwin = (i >= 0) & (T["t"] - ep_t[np.maximum(i, 0)] <= 10 * S)
            against = inwin & (T["side"] == -ep_d[np.maximum(i, 0)])
            withf = inwin & ~against
            r["liquidation"] = {"against": stats(T, against), "with": stats(T, withf), "outside": stats(T, ~inwin)}
        report["strategies"][name] = r

    tag = f"{days[0]}_{days[-1]}"
    out = Path("data/results") / f"research_queue_{tag}.json"
    out.write_text(json.dumps(report))

    for name, r in report["strategies"].items():
        a = r["all"]
        print(f"\n{name}: {a['n']:,} fills, markout 1s {a['markout_1s_bps']:+.3f}  5s {a['markout_5s_bps']:+.3f}  30s {a['markout_30s_bps']:+.3f} bp")
        print(f"  {'how the fill happened':<34}{'share':>7}{'1s':>9}{'5s':>9}{'30s':>9}")
        for how, s in r["how"].items():
            if s["n"]:
                print(f"  {how:<34}{s['share']:>7.0%}{s['markout_1s_bps']:>+9.3f}{s['markout_5s_bps']:>+9.3f}{s['markout_30s_bps']:>+9.3f}")
        print(f"  {'volume ahead at entry (BTC)':<34}{'n':>7}{'1s':>9}{'5s':>9}{'30s':>9}{'queue fills':>13}")
        for s in r["ahead"]:
            lab = f"{s['range_btc'][0]:.2f} to {s['range_btc'][1]:.2f}"
            print(f"  {lab:<34}{s['n']:>7}{s['markout_1s_bps']:>+9.3f}{s['markout_5s_bps']:>+9.3f}{s['markout_30s_bps']:>+9.3f}{s['share_queue_fills']:>13.0%}")
        print(f"  {'time in queue before the fill':<34}{'n':>7}{'1s':>9}{'5s':>9}{'30s':>9}")
        for s in r["wait"]:
            if s["n"]:
                lab = f"{s['range_s'][0]:g} to {s['range_s'][1]:g} s" if s["range_s"][1] < 1e8 else f"over {s['range_s'][0]:g} s"
                print(f"  {lab:<34}{s['n']:>7}{s['markout_1s_bps']:>+9.3f}{s['markout_5s_bps']:>+9.3f}{s['markout_30s_bps']:>+9.3f}")
        if r["liquidation"]:
            print(f"  {'10 s after a liquidation episode':<34}{'n':>7}{'1s':>9}{'5s':>9}{'30s':>9}")
            for k, lab in (("against", "forced flow trading against us"), ("with", "forced flow on our side"), ("outside", "no liquidation in last 10 s")):
                s = r["liquidation"][k]
                if s["n"]:
                    print(f"  {lab:<34}{s['n']:>7}{s['markout_1s_bps']:>+9.3f}{s['markout_5s_bps']:>+9.3f}{s['markout_30s_bps']:>+9.3f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
