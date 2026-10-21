"""Liquidations as forced order flow: what they do to the price, on which venue, and to a market maker.

Usage:
  python -m research.liquidations --dates 2026-10-15,2026-10-16,2026-10-17,2026-10-18

Needs the rebuilt books (python -m rebuild) and the extracted events (python -m research.events).

1. Checks.  Each venue's side convention against the aggressor side of that venue's own
   trades within 200 ms of the liquidation (a forced sell should print as seller-initiated).
2. Episodes.  BTC liquidations on all three venues, grouped into episodes: a new episode
   starts after 2 s without a liquidation anywhere. Direction = sign of the net forced notional.
3. Price path.  The Binance mid (real-time best bid and offer) from 30 s before to 120 s after
   each episode starts, in bp, signed by the episode's direction, so positive = moved the way
   the forced flow pushed.
4. Forced against voluntary flow.  Is forced flow less informed than ordinary aggressive flow of
   the same size? Both are measured the same way: net aggressive notional on Binance in the 2 s
   from the start. Controls are 2 s windows at least 60 s from any liquidation, matched to each
   episode tercile on that measure. Retention = move at +120 s over the peak move in the first
   10 s: below 1 means the push partly reverses.
5. Venues.  Signed mid paths on Binance, Bybit and OKX around episodes, at 50 ms resolution, and
   which venue printed the first liquidation of the episode. Exchange clocks differ by a few ms,
   so sub-10 ms differences are not interpreted.

Writes data/results/research_liquidations_<first>_<last>.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

S = 1_000_000_000
GRID = np.r_[np.arange(-30, -5, 1.0), np.arange(-5, 10, 0.25), np.arange(10, 121, 2.0)]
FINE = np.arange(-2, 5.001, 0.05)
GAP_S = 2.0
FLOW_S = 2


def load_mid(clean: Path, venue: str, symbol: str, day: str):
    base = clean / venue / symbol / day
    if venue == "binance" and (base / "bbo.parquet").exists():
        t = pq.read_table(base / "bbo.parquet", columns=["exch_ns", "bid_px", "ask_px"])
        x, b, a = (t.column(c).to_numpy() for c in ("exch_ns", "bid_px", "ask_px"))
    else:
        t = pq.read_table(base / "top.parquet", columns=["exch_ns", "bid_px_1", "ask_px_1"])
        x, b, a = (t.column(c).to_numpy() for c in ("exch_ns", "bid_px_1", "ask_px_1"))
    o = np.argsort(x, kind="stable")
    m = 0.5 * (b[o] + a[o])
    ok = np.isfinite(m) & (m > 0)
    return x[o][ok].astype(np.int64), m[ok]


def load_trades(clean: Path, venue: str, symbol: str, day: str):
    t = pq.read_table(clean / venue / symbol / day / "trades.parquet", columns=["exch_ns", "price", "qty", "side"])
    x = t.column("exch_ns").to_numpy()
    o = np.argsort(x, kind="stable")
    return x[o].astype(np.int64), t.column("price").to_numpy()[o], t.column("qty").to_numpy()[o], t.column("side").to_numpy()[o]


def at(ts: np.ndarray, vals: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Last value at or before each query time (NaN before the first)."""
    i = np.searchsorted(ts, q, side="right") - 1
    out = np.full(q.shape, np.nan)
    ok = i >= 0
    out[ok] = vals[i[ok]]
    return out


def signed_paths(ts, mid, t0, sign, grid):
    q = t0[:, None] + (grid[None, :] * S).astype(np.int64)
    m = at(ts, mid, q)
    base = at(ts, mid, t0 - 1)               # the mid just before the episode
    return sign[:, None] * (m / base[:, None] - 1.0) * 1e4


def episodes(liq: dict) -> list[dict]:
    t, side, usd, venue = liq["t"], liq["side"], liq["usd"], liq["venue"]
    o = np.argsort(t, kind="stable")
    t, side, usd, venue = t[o], side[o], usd[o], venue[o]
    starts = np.r_[0, np.flatnonzero(np.diff(t) > GAP_S * S) + 1]
    ends = np.r_[starts[1:], len(t)]
    eps = []
    for a, b in zip(starts, ends):
        net = float(np.sum(side[a:b] * usd[a:b]))
        if net == 0:
            continue
        eps.append({"t0": int(t[a]), "dir": 1 if net > 0 else -1, "usd": float(np.sum(usd[a:b])), "n": int(b - a),
                    "venues": sorted(set(venue[a:b].tolist())), "first": str(venue[a])})
    return eps


def flow_per_second(tt, tp, tq, ts, day_start):
    """Net aggressive notional per second over the day (USD, + = buyer-initiated)."""
    sec = ((tt - day_start) // S).astype(np.int64)
    ok = (sec >= 0) & (sec < 86400)
    f = np.zeros(86400 + FLOW_S)
    np.add.at(f, sec[ok], ts[ok] * tp[ok] * tq[ok])
    return f


def bootstrap_ci(x: np.ndarray, fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < 5:
        return [float("nan"), float("nan")]
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def retention(paths: np.ndarray) -> float:
    mean = np.nanmean(paths, axis=0)
    early = (GRID >= 0) & (GRID <= 10)
    peak = np.nanmax(mean[early])
    late = mean[np.argmin(np.abs(GRID - 120))]
    return float(late / peak) if peak > 0 else float("nan")


def summary(paths: np.ndarray) -> dict:
    mean = np.nanmean(paths, axis=0)
    i = lambda s: int(np.argmin(np.abs(GRID - s)))   # noqa: E731
    early = (GRID >= 0) & (GRID <= 10)
    return {"n": int(len(paths)), "move_1s": float(mean[i(1)]), "move_5s": float(mean[i(5)]), "peak_10s": float(np.nanmax(mean[early])),
            "move_60s": float(mean[i(60)]), "move_120s": float(mean[i(120)]), "pre_30s": float(mean[i(-30)]),
            "retention": retention(paths),
            "retention_ci": bootstrap_ci(paths, retention), "path": [round(float(v), 4) for v in mean]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--max-controls", type=int, default=3000)
    args = ap.parse_args()
    days = [d.strip() for d in args.dates.split(",") if d.strip()]
    clean = Path(args.clean)
    rng = np.random.default_rng(1)

    sym_book = {"binance": "BTCUSDT", "bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP"}
    checks = Counter()
    all_eps, ep_paths, ep_flow = [], [], []
    ctl_paths, ctl_flow = [], []
    venue_paths = {v: [] for v in sym_book}
    first_counts = Counter()

    for day in days:
        tab = pq.read_table(clean / "events" / day / "liquidations.parquet")
        c = {k: tab.column(k).to_numpy(zero_copy_only=False) for k in tab.column_names}
        keep = (c["symbol"] == "BTC") & (c["exch_ns"] > 0) & (c["side"] != 0) & (c["qty"] > 0) & (c["price"] > 0)
        lq = {k: v[keep] for k, v in c.items()}
        if not keep.any():
            print(f"{day}: no BTC liquidations")
            continue
        # 1. side convention check against each venue's own trades
        for v in sorted(set(lq["venue"].tolist())):
            g = lq["venue"] == v
            try:
                tt, tp, tq, tsd = load_trades(clean, v, sym_book[v], day)
            except FileNotFoundError:
                continue
            for t, s in zip(lq["exch_ns"][g], lq["side"][g]):
                a, b = np.searchsorted(tt, [t - 200_000_000, t + 200_000_000])
                if b > a:
                    net = float(np.sum(tsd[a:b] * tq[a:b]))
                    if net != 0:
                        checks[(v, "agree" if np.sign(net) == s else "disagree")] += 1
        liq = {"t": lq["exch_ns"].astype(np.int64), "side": lq["side"].astype(int),
               "usd": lq["qty"] * lq["price"], "venue": lq["venue"]}
        eps = episodes(liq)
        bx, bm = load_mid(clean, "binance", "BTCUSDT", day)
        tt, tp, tq, tsd = load_trades(clean, "binance", "BTCUSDT", day)
        day_start = int(tt[0] // (86400 * S) * 86400 * S) if len(tt) else int(np.datetime64(day, "ns").astype(np.int64))
        flow = flow_per_second(tt, tp, tq, tsd, day_start)
        t0 = np.array([e["t0"] for e in eps], dtype=np.int64)
        sign = np.array([e["dir"] for e in eps], dtype=float)
        sec = np.clip((t0 - day_start) // S, 0, 86399)
        fl = np.array([flow[s:s + FLOW_S].sum() for s in sec])
        ep_paths.append(signed_paths(bx, bm, t0, sign, GRID))
        ep_flow.append(np.abs(fl))
        for e, f in zip(eps, fl):
            e["flow_usd"] = float(f)
            e["day"] = day
            first_counts[e["first"]] += 1
        all_eps += eps
        for v in sym_book:
            try:
                vx, vm = load_mid(clean, v, sym_book[v], day)
                venue_paths[v].append(signed_paths(vx, vm, t0, sign, FINE))
            except FileNotFoundError:
                pass
        # controls: 2 s windows at least 60 s from any liquidation, with real aggressive flow
        near = np.zeros(86400 + FLOW_S, dtype=bool)
        for s in sec:
            near[max(0, s - 60):s + 61] = True
        win = np.convolve(flow, np.ones(FLOW_S), mode="full")[FLOW_S - 1:FLOW_S - 1 + 86400]
        cand = np.flatnonzero(~near[:86400] & (np.abs(win) > 0))
        cand = cand[(cand > 30) & (cand < 86400 - 121)]
        if len(cand):
            pick = rng.choice(cand, size=min(len(cand), args.max_controls * 3), replace=False)
            ct0 = day_start + pick.astype(np.int64) * S
            ctl_paths.append(signed_paths(bx, bm, ct0, np.sign(win[pick]), GRID))
            ctl_flow.append(np.abs(win[pick]))
        print(f"{day}: {int(keep.sum()):,} BTC liquidations in {len(eps):,} episodes")

    P = np.vstack(ep_paths)
    F = np.concatenate(ep_flow)
    CP = np.vstack(ctl_paths) if ctl_paths else np.zeros((0, len(GRID)))
    CF = np.concatenate(ctl_flow) if ctl_flow else np.zeros(0)
    edges = np.quantile(F, [0, 1 / 3, 2 / 3, 1])
    report = {"dates": days, "grid_s": GRID.tolist(), "fine_grid_s": FINE.round(3).tolist(),
              "episodes": len(all_eps), "liquidations": int(sum(e["n"] for e in all_eps)),
              "usd_total": float(sum(e["usd"] for e in all_eps)),
              "side_check": {f"{v}": {"agree": checks[(v, 'agree')], "disagree": checks[(v, 'disagree')]} for v in sym_book},
              "first_venue": dict(first_counts), "all": summary(P), "terciles": []}
    for k in range(3):
        lo, hi = edges[k], edges[k + 1]
        m = (F >= lo) & (F <= hi)
        cm = (CF >= lo) & (CF <= hi)
        cidx = np.flatnonzero(cm)
        if len(cidx) > args.max_controls:
            cidx = rng.choice(cidx, args.max_controls, replace=False)
        report["terciles"].append({"flow_usd_range": [float(lo), float(hi)], "forced": summary(P[m]),
                                   "voluntary": summary(CP[cidx]) if len(cidx) else None})
    report["venues"] = {v: [round(float(x), 4) for x in np.nanmean(np.vstack(p), axis=0)] for v, p in venue_paths.items() if p}
    # how long until each venue's mean path first reaches half its 5 s move
    report["venue_half_time_s"] = {}
    for v, path in report["venues"].items():
        path = np.array(path)
        target = 0.5 * path[np.argmin(np.abs(FINE - 5))]
        hit = np.flatnonzero((FINE >= 0) & (path >= target)) if target > 0 else []
        report["venue_half_time_s"][v] = float(FINE[hit[0]]) if len(hit) else None
    report["episode_list"] = [{k: e[k] for k in ("day", "t0", "dir", "usd", "n", "first", "flow_usd")} for e in all_eps]

    tag = f"{days[0]}_{days[-1]}"
    out = Path("data/results") / f"research_liquidations_{tag}.json"
    out.write_text(json.dumps(report))

    print(f"\n{report['liquidations']:,} BTC liquidations, {report['episodes']:,} episodes, ${report['usd_total'] / 1e6:.1f}m forced notional")
    print("Side convention check (share of liquidations matching the aggressor side of that venue's trades within 200 ms):")
    for v, c in report["side_check"].items():
        n = c["agree"] + c["disagree"]
        print(f"  {v:<8} {c['agree'] / n:.0%} of {n:,}" if n else f"  {v:<8} no data")
    print("First venue to print a liquidation in each episode:", dict(first_counts))
    a = report["all"]
    print(f"\nAll episodes, Binance mid in the forced direction (bp): pre-30s {a['pre_30s']:+.2f}, +1s {a['move_1s']:+.2f}, "
          f"+5s {a['move_5s']:+.2f}, peak {a['peak_10s']:+.2f}, +60s {a['move_60s']:+.2f}, +120s {a['move_120s']:+.2f}")
    print("\nForced against voluntary flow of the same size (net aggressive Binance flow in the first 2 s):")
    print(f"{'flow tercile ($)':<26}{'':10}{'n':>7}{'+1s':>8}{'peak':>8}{'+120s':>8}{'retention':>11}{'95% CI':>18}")
    for tq in report["terciles"]:
        r = f"{tq['flow_usd_range'][0] / 1e3:,.0f}k to {tq['flow_usd_range'][1] / 1e3:,.0f}k"
        for kind in ("forced", "voluntary"):
            s = tq[kind]
            if s is None:
                continue
            ci = s["retention_ci"]
            print(f"{r:<26}{kind:<10}{s['n']:>7}{s['move_1s']:>+8.2f}{s['peak_10s']:>+8.2f}{s['move_120s']:>+8.2f}"
                  f"{s['retention']:>11.2f}{f'[{ci[0]:.2f}, {ci[1]:.2f}]':>18}")
            r = ""
    print("\nTime for each venue's mid to make half its 5 s move after the episode starts:",
          {v: (f"{t:.2f}s" if t is not None else "-") for v, t in report["venue_half_time_s"].items()})
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
