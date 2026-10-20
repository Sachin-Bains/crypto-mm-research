"""Pool frontier runs across days: one point per method and dial.

Usage:  python tools/pool_frontier.py --dates 2026-10-21,2026-10-22

Volume = total fills and notional over the days. Adverse selection = notional-weighted
5 s markout per unit traded. For the conformal quoter the realised loss is also
compared with its budget, which is the guarantee being tested.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--results", default="data/results")
    args = ap.parse_args()
    acc = defaultdict(lambda: {"fills": 0, "notional": 0.0, "m5": 0.0, "a1": 0.0, "gross": 0.0, "rebate": 0.0,
                               "real": 0.0, "w": 0.0})
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        data = json.loads((Path(args.results) / f"frontier_{d}.json").read_text())
        for p in data["points"]:
            a = acc[(p["method"], p["dial"])]
            n = p["notional_usd"] or 0.0
            a["fills"] += p["fills"]
            a["notional"] += n
            a["m5"] += (p.get("markout_5s_bps") or 0.0) * n
            a["a1"] += (p.get("adverse_1s_bps") or 0.0) * n
            a["gross"] += p.get("pnl_gross_usd") or 0.0
            a["rebate"] += p.get("pnl_net_usd [MM rebate (-0.5 bp)]") or 0.0
            if "conformal" in p:
                c = p["conformal"]
                a["real"] += c["realised_loss_bp"] * c["settled_fills"]
                a["w"] += c["settled_fills"]
    print(f"{'method':<16}{'dial':>7}{'fills':>9}{'notional $m':>13}{'markout 5s bp':>15}{'adverse 1s bp':>15}"
          f"{'gross $':>10}{'w/ rebate $':>13}{'budget check':>16}")
    for (m, dial), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1] if kv[0][1] is not None else -1)):
        n = max(a["notional"], 1e-12)
        chk = f"{a['real'] / a['w']:.3f} vs {dial:.2f}" if a["w"] else ""
        print(f"{m:<16}{str(dial if dial is not None else '-'):>7}{a['fills']:>9}{a['notional'] / 1e6:>13.1f}"
              f"{a['m5'] / n:>+15.3f}{a['a1'] / n:>+15.3f}{a['gross']:>10.0f}{a['rebate']:>13.0f}{chk:>16}")


if __name__ == "__main__":
    main()
