"""Calibrate the Avellaneda-Stoikov order-arrival curve on development days.

Usage:  python tools/calibrate_as.py --dates 2026-10-15,2026-10-16,2026-10-17,2026-10-18

The model assumes market orders reach a quote delta bp from the mid at rate
    lambda(delta) = A * exp(-k * delta)   per second.
For each second and each side, take the mid at the start of the second (real-time best bid and
offer) and the furthest trade price reached on that side during the second. lambda(delta) is
the share of seconds in which trades reached at least delta, counted on both sides and averaged.
A and k come from a straight-line fit of ln(lambda) on delta over distances from one tick out to
where fewer than 1 in 10,000 seconds reach. Development days only, so the test days stay unseen.
Writes data/results/as_calibration.json, which sim.frontier reads for the "as" method.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.data import load_day  # noqa: E402

S = 1_000_000_000


def reach(md) -> tuple[np.ndarray, float]:
    """Per-second furthest reach in bp on each side (buy side above mid, sell side below), and
    the median mid."""
    if not md.has_bbo or not len(md.t_exch):
        return np.zeros(0), np.nan
    sec = md.t_exch // S
    t0 = sec * S
    i = np.searchsorted(md.b_exch, t0, side="right") - 1
    ok = i >= 0
    mid = np.full(len(sec), np.nan)
    mid[ok] = 0.5 * (md.b_bid[i[ok]] + md.b_ask[i[ok]])
    d = (md.t_price - mid) / mid * 1e4 * md.t_side    # buyer: above mid; seller: below mid; positive = away
    good = np.isfinite(d)
    sec, side, d = sec[good], md.t_side[good], d[good]
    out = []
    for s in (1, -1):
        m = side == s
        if not m.any():
            continue
        us, inv = np.unique(sec[m], return_inverse=True)
        mx = np.full(len(us), -np.inf)
        np.maximum.at(mx, inv, d[m])
        out.append(mx)
    # seconds with no trade on a side reach nothing: count them as -inf
    n_sec = int(md.t_exch[-1] // S - md.t_exch[0] // S + 1)
    allr = []
    for mx in out:
        pad = np.full(max(0, n_sec - len(mx)), -np.inf)
        allr.append(np.r_[mx, pad])
    return np.concatenate(allr) if allr else np.zeros(0), float(np.nanmedian(mid))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--tick", type=float, default=0.1)
    args = ap.parse_args()
    rs, mids = [], []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        md = load_day(args.clean, "binance", "BTCUSDT", d)
        r, m = reach(md)
        rs.append(r)
        mids.append(m)
        print(f"{d}: {len(r) // 2} seconds, median mid {m:,.1f}")
    r = np.concatenate(rs)
    tick_bp = args.tick / float(np.nanmedian(mids)) * 1e4
    grid = tick_bp * np.arange(1, 200)
    lam = np.array([(r >= g).mean() for g in grid])
    use = lam >= 1e-4
    x, y = grid[use], np.log(lam[use])
    slope, icept = np.polyfit(x, y, 1)
    k, A = -slope, float(np.exp(icept))
    out = {"dates": args.dates, "tick_bp": round(tick_bp, 5), "k_per_bp": round(k, 3), "A_per_s": round(A, 4),
           "fit_points": int(use.sum()), "fit_max_bp": round(float(x.max()), 4),
           "curve": [[round(float(g), 4), round(float(v), 6)] for g, v in zip(grid[:60], lam[:60])]}
    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/as_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"tick = {tick_bp:.4f} bp; lambda(delta) = {A:.3f} * exp(-{k:.2f} * delta) per second, "
          f"fitted on {use.sum()} points up to {x.max():.3f} bp")
    print("Saved to data/results/as_calibration.json")


if __name__ == "__main__":
    main()
