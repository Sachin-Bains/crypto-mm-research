"""Pool the out-of-sample Layer 4 runs across days, for the pre-registered test.

Usage:  python tools/pool_oos.py              (reads data/results/layer4_*_oos.json)
        python tools/pool_oos.py --label oos --test full_pull --ref random_off_pull

Prints one line per day and strategy, then the pooled test: the hourly 5 s markout of
--test minus --ref over every hour of every day (paired, hours as blocks), and per-fill
numbers over all fills. Settings and the test itself were fixed in the README before
any of these days were used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.analysis import paired_vs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results")
    ap.add_argument("--label", default="oos")
    ap.add_argument("--test", default="full_pull")
    ap.add_argument("--ref", default="random_off_pull")
    args = ap.parse_args()

    files = sorted(Path(args.results).glob(f"layer4_*_{args.label}.json"))
    if not files:
        raise SystemExit(f"No layer4_*_{args.label}.json files in {args.results}")

    hourly = {args.test: {}, args.ref: {}}
    tot = {}
    wins = days = 0
    print(f"{'date':<12}{'strategy':<18}{'fills':>8}{'gross_bps':>11}{'adverse_1s':>12}{'gross_usd':>12}{'t_vs_ref':>10}")
    for f in files:
        d = json.loads(f.read_text())
        date = d["args"]["date"]
        rows = {r["strategy"]: r for r in d["results"]}
        for name in (args.ref, args.test, "join"):
            r = rows.get(name)
            if r is None:
                continue
            t = r.get("t_vs_ref")
            print(f"{date:<12}{name:<18}{r['fills']:>8}{r.get('pnl_gross_bps', 0):>+11.3f}"
                  f"{r.get('adverse_1s_bps', float('nan')):>+12.3f}{r.get('pnl_gross_usd', 0):>+12.2f}"
                  f"{(f'{t:+.1f}' if isinstance(t, float) and t == t else '-'):>10}")
            if name in hourly:
                hourly[name].update({int(k): v for k, v in r.get("hourly_markout5s_usd", {}).items()})
                a = tot.setdefault(name, [0.0, 0.0, 0])
                a[0] += r.get("pnl_gross_usd", 0.0)
                a[1] += r.get("notional_usd", 0.0)
                a[2] += r["fills"]
        if args.test in rows and args.ref in rows:
            days += 1
            wins += rows[args.test]["pnl_gross_bps"] > rows[args.ref]["pnl_gross_bps"]
        print()

    print(f"Pooled over {days} day(s):")
    for name, (g, n, k) in tot.items():
        print(f"  {name:<18} fills {k:>8}   gross {g:>+10.2f} USD   {g / max(n, 1e-12) * 1e4:+.3f} bp per unit traded")
    diff, t, nh = paired_vs(hourly[args.ref], hourly[args.test])
    print(f"  {args.test} minus {args.ref}: {diff:+.2f} USD per hour (5 s markout), t = {t:+.2f} over {nh} hours")
    print(f"  days on which {args.test} was better per fill: {wins} of {days}")


if __name__ == "__main__":
    main()
