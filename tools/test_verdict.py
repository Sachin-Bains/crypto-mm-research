"""Pre-registered verdicts for versions 2 and 3 on the test days.

Usage:
    python tools/test_verdict.py --dates 2026-10-22,2026-10-23 --out data/results/verdict_primary.json
    python tools/test_verdict.py --dates 2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28 --out ...

Readings, exactly as written in the README before the test:
  v2 claim 1 (budget): for every budget where lam sat on a bound for under 5% of updates,
      pooled realised 5 s loss is within 0.05 bp of B. Judged on the warm start.
  v2 claim 2 (frontier): at the notional of each conformal point, its 5 s loss per unit is
      below the volatility filter's, interpolated along the volatility-filter curve.
  v3 claim 1: at the notional of each conformal_q point with B = 0.1 and 0.2, its 5 s loss per
      unit is below version 2's, interpolated along version 2's curve.
  v3 claim 2: the budget check applies unchanged.

Controls come from the plain (cold) files, which hold every method; conformal methods are read
from the warm files for the warm reading and from the plain files for the cold one.

Where a point's notional lies outside the reference curve, the curve is extended flat from its
nearest end (stated in the output as 'extrapolated'), and the strict check against the reference
curve's best point anywhere is reported beside it. Uncertainty: hour-block bootstrap (hours
resampled with replacement, the same hours for both methods), 2000 draws.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CONF = ("conformal", "conformal_q")


def load(results: Path, dates: list[str]):
    """Return {(reading, method, dial): {'hours': {h: (markout_usd, notional)}, 'real': .., 'w': .., 'bound': ..}}."""
    out: dict = defaultdict(lambda: {"hours": defaultdict(lambda: [0.0, 0.0]), "real": 0.0, "w": 0.0, "bw": 0.0})
    for d in dates:
        for label, reading in (("", "cold"), ("_warm", "warm")):
            data = json.loads((results / f"frontier_{d}{label}.json").read_text())
            for p in data["points"]:
                m = p["method"]
                if reading == "warm" and m not in CONF:
                    continue
                keys = [(reading, m, p["dial"])]
                if reading == "cold" and m not in CONF:
                    keys = [("ctrl", m, p["dial"])]
                for k in keys:
                    a = out[k]
                    for h, v in p["hourly_markout5s_usd"].items():
                        a["hours"][h][0] += v
                        a["hours"][h][1] += p["hourly_notional_usd"].get(h, 0.0)
                    if "conformal" in p:
                        c = p["conformal"]
                        a["real"] += c["realised_loss_bp"] * c["settled_fills"]
                        a["w"] += c["settled_fills"]
                        a["bw"] += c["share_updates_at_bound"] * c["settled_fills"]
    return out


def point(a, hours=None):
    hs = a["hours"]
    if hours is None:
        m = sum(v[0] for v in hs.values()); n = sum(v[1] for v in hs.values())
    else:
        m = sum(hs[h][0] if h in hs else 0.0 for h in hours)
        n = sum(hs[h][1] if h in hs else 0.0 for h in hours)
    return n, (-m / n * 1e4 if n > 0 else np.nan)


def curve(pts):
    pts = sorted(p for p in pts if p[0] > 0)
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def interp(x, xs, ys):
    return float(np.interp(x, xs, ys)), bool(x < xs[0] or x > xs[-1])


def compare(data, reading, method, dial, ref_method, rng, draws=2000, ref_reading="ctrl"):
    a = data[(reading, method, dial)]
    refs = [k for k in data if k[0] == ref_reading and k[1] == ref_method]
    n, loss = point(a)
    xs, ys = curve([point(data[k]) for k in refs])
    ref, extrap = interp(n, xs, ys)
    hours = sorted(set().union(*[set(data[k]["hours"]) for k in refs + [(reading, method, dial)]]))
    diffs = []
    for _ in range(draws):
        hb = list(rng.choice(hours, size=len(hours), replace=True))
        nb, lb = point(a, hb)
        xb, yb = curve([point(data[k], hb) for k in refs])
        if len(xb) < 2 or not np.isfinite(lb):
            continue
        diffs.append(lb - interp(nb, xb, yb)[0])
    diffs = np.array(diffs)
    return {
        "notional_m": round(n / 1e6, 2), "loss_bp": round(loss, 3),
        "ref_loss_bp": round(ref, 3), "extrapolated": extrap,
        "diff_bp": round(loss - ref, 3),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 3), round(float(np.percentile(diffs, 97.5)), 3)],
        "p_not_below": round(float((diffs >= 0).mean()), 4),
        "ref_best_bp": round(float(ys.min()), 3),
        "pass": loss < ref,
    }


def budget(data, reading, method, dial):
    a = data[(reading, method, dial)]
    real = a["real"] / a["w"]; bound = a["bw"] / a["w"]
    applies = bound < 0.05
    return {"realised_bp": round(real, 3), "budget": dial, "gap_bp": round(real - dial, 3),
            "share_at_bound": round(bound, 4), "applies": applies,
            "pass": (abs(real - dial) <= 0.05) if applies else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--results", default="data/results")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    data = load(Path(args.results), dates)
    rng = np.random.default_rng(7)
    budgets = (0.1, 0.2, 0.3, 0.4)
    res = {"dates": dates, "hours": len(set().union(*[set(v["hours"]) for v in data.values()])), "readings": {}}

    ctrl = {}
    for (r, m, dial), a in data.items():
        if r == "ctrl":
            n, l = point(a)
            ctrl.setdefault(m, []).append({"dial": dial, "notional_m": round(n / 1e6, 2), "loss_bp": round(l, 3)})
    res["controls"] = ctrl

    for reading in ("warm", "cold"):
        R = {"points": {}, "v2_budget": {}, "v2_frontier": {}, "v3_vs_v2": {}, "v3_budget": {}}
        for m in CONF:
            R["points"][m] = [dict(zip(("dial", "notional_m", "loss_bp"),
                                       (b, round(point(data[(reading, m, b)])[0] / 1e6, 2),
                                        round(point(data[(reading, m, b)])[1], 3)))) for b in budgets]
        for b in budgets:
            R["v2_budget"][b] = budget(data, reading, "conformal", b)
            R["v3_budget"][b] = budget(data, reading, "conformal_q", b)
            R["v2_frontier"][b] = compare(data, reading, "conformal", b, "vol_pull", rng)
        for b in (0.1, 0.2):
            R["v3_vs_v2"][b] = compare(data, reading, "conformal_q", b, "conformal", rng, ref_reading=reading)
        res["readings"][reading] = R

    txt = json.dumps(res, indent=1)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt)


if __name__ == "__main__":
    main()
