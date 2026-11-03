"""Export short windows of the test days for the interactive replay: the order book, trades,
liquidations, and what each strategy quoted, when it was filled, and what each fill was worth.

Usage:
  python tools/export_replay.py --dates 2026-10-22,2026-10-23,2026-10-24,2026-10-25,2026-10-26,2026-10-27,2026-10-28 \
      --intervals-dates 2026-10-14,2026-10-28 --auto

  --auto picks three 10-minute windows across the dates:
     cascade  the largest burst of BTC liquidations (all three venues, net USD in 60 s), from 3 minutes
              before it to 7 minutes after
     busy     the 10 minutes with the most Binance BTC volume traded
     normal   a 10 minutes whose volume is the median of all valid windows
  --window NAME=YYYY-MM-DDTHH:MM adds a window by hand (10 minutes from that UTC time).

Each strategy is replayed exactly as in the test run (same simulator, same 150 ms latency, version 2 and
3 thresholds warm-started from the same files), from the start of the day to the end of the window, so
its state on entering the window is the real one. Only events inside the window are written.

Output: data/results/replay/replay_<name>.json (times in ms from the window start, prices in ticks
from a base price) and data/results/replay/index.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.analysis import mid_at  # noqa: E402
from sim.engine import BID, QUEUE, Engine  # noqa: E402
from sim.frontier import load_signal  # noqa: E402
from sim.strategies import VARIANTS, ConformalBudgetQuoter, ForecastQuoter, V3Quoter  # noqa: E402

MS = 1_000_000
S = 1_000_000_000
TICK = 0.1
LEVELS = 20
FRAME_MS = 100
WIN_MIN = 10
LAT_MS = 150

STRATS = {
    # name: (label, method, dial)
    "v3": ("Version 3, B 0.1", "conformal_q", 0.1),
    "v2": ("Version 2, B 0.2", "conformal", 0.2),
    "vol": ("Volatility filter, 0.8", "vol_pull", 0.8),
}


class RecEngine(Engine):
    """The simulator, unchanged, plus a log of our orders, fills and book frames inside a window."""

    def __init__(self, *a, w0: int, w1: int, base: int, **kw) -> None:
        super().__init__(*a, **kw)
        self.w0, self.w1, self.base = w0, w1, base
        self.ev: list = []
        self.frames: list = []
        self.next_frame = w0
        self.snapped = False

    def _in(self, t: int) -> bool:
        return self.w0 <= t < self.w1

    def _tk(self, px: float) -> int:
        return int(round(px / TICK)) - self.base

    def _snap(self, t: int) -> None:
        if not self.snapped and t >= self.w0:
            self.snapped = True
            for o in self.live.values():
                self.ev.append([0, "live", o.side, self._tk(o.price), o.oid, round(o.ahead, 4)])

    def _on_place(self, t, o):
        self._snap(t)
        super()._on_place(t, o)
        if self._in(t):
            self.ev.append([(t - self.w0) // MS, "place" if o.status == "live" else "reject", o.side,
                            self._tk(o.price), o.oid, round(o.ahead0, 4)])

    def _on_cancel(self, t, oid):
        self._snap(t)
        was = oid in self.live
        super()._on_cancel(t, oid)
        if was and self._in(t):
            self.ev.append([(t - self.w0) // MS, "cancel", 0, 0, oid, 0])

    def _cancel_all(self, t, reason):
        self._snap(t)
        ids = list(self.live)
        super()._cancel_all(t, reason)
        if self._in(t):
            for oid in ids:
                self.ev.append([(t - self.w0) // MS, "cancel", 0, 0, oid, 0])

    def _fill(self, o, x, t, how=""):
        self._snap(t)
        n0 = len(self.res.fills)
        super()._fill(o, x, t, how)
        if self._in(t) and len(self.res.fills) > n0:
            self.ev.append([(t - self.w0) // MS, "fill", o.side, self._tk(o.price), o.oid, round(x, 4), how])

    def _on_depth(self, g):
        super()._on_depth(g)
        t = self.now
        if self.valid and self._in(t) and t >= self.next_frame and self.bids and self.asks:
            b = [[self._tk(p), round(q, 3)] for p, q in reversed(self.bids.items()[-LEVELS:])]
            a = [[self._tk(p), round(q, 3)] for p, q in self.asks.items()[:LEVELS]]
            self.frames.append([(t - self.w0) // MS, b, a])
            self.next_frame = t - (t - self.w0) % (FRAME_MS * MS) + FRAME_MS * MS


def lam0_for(date: str, method: str, dial: float) -> float:
    """The threshold the test run started this day with (warm start)."""
    f = Path("data/results") / f"frontier_{date}_warm.json"
    if not f.exists():
        return 3.0
    src = json.loads(f.read_text())["args"].get("lam0_from")
    if not src:
        return 3.0
    prev = Path("data/results") / f"frontier_{src}.json"
    if not prev.exists():
        return 3.0
    for p in json.loads(prev.read_text())["points"]:
        if p["method"] == method and p["dial"] == dial and "conformal" in p:
            return p["conformal"]["lam_final"]
    return 3.0


def liquidations(clean: str, dates: list[str]):
    ts, sd, usd, ven = [], [], [], []
    for d in dates:
        f = Path(clean) / "events" / d / "liquidations.parquet"
        if not f.exists():
            continue
        t = pq.read_table(f)
        c = {k: t.column(k).to_numpy(zero_copy_only=False) for k in t.column_names}
        k = (c["symbol"] == "BTC") & (c["exch_ns"] > 0) & (c["qty"] > 0) & (c["price"] > 0) & (c["side"] != 0)
        ts.append(c["exch_ns"][k]); sd.append(c["side"][k]); usd.append(c["qty"][k] * c["price"][k]); ven.append(c["venue"][k])
    if not ts:
        return None
    t = np.concatenate(ts); o = np.argsort(t)
    return t[o], np.concatenate(sd)[o], np.concatenate(usd)[o], np.concatenate(ven)[o]


def valid_windows(md, step_ns: int):
    """Start times of 10-minute windows with a valid book throughout, and their traded notional."""
    if not len(md.g_start):
        return np.zeros(0, np.int64), np.zeros(0)
    gt = md.d_exch[md.g_start]
    kinds = md.d_kind[md.g_start]
    bad = gt[kinds == 2]
    starts = np.arange((gt[0] // step_ns + 1) * step_ns, gt[-1] - WIN_MIN * 60 * S, step_ns)
    ok = np.array([not ((bad >= s - 60 * S) & (bad < s + WIN_MIN * 60 * S)).any() for s in starts], bool)
    starts = starts[ok]
    cum = np.r_[0.0, np.cumsum(md.t_price * md.t_qty)]
    lo = np.searchsorted(md.t_exch, starts); hi = np.searchsorted(md.t_exch, starts + WIN_MIN * 60 * S)
    return starts, cum[hi] - cum[lo]


def pick_auto(args, dates):
    from sim.data import load_day
    cands = []
    for d in dates:
        md = load_day(args.clean, "binance", "BTCUSDT", d)
        st, vol = valid_windows(md, 60 * S)
        cands += [(int(s), float(v), d) for s, v in zip(st, vol)]
        print(f"  {d}: {len(st)} valid windows")
    if not cands:
        raise SystemExit("no valid windows")
    out = {}
    busy = max(cands, key=lambda c: c[1])
    out["busy"] = (busy[2], busy[0])
    vols = np.array([c[1] for c in cands])
    med = cands[int(np.argmin(np.abs(vols - np.median(vols))))]
    out["normal"] = (med[2], med[0])
    liq = liquidations(args.clean, dates)
    if liq is not None and len(liq[0]):
        t, sd, usd, _ = liq
        win = 60 * S
        cum = np.r_[0.0, np.cumsum(usd)]
        j = np.searchsorted(t, t + win)
        burst = cum[j] - cum[np.arange(len(t))]
        valid = {(c[2], c[0]) for c in cands}
        order = np.argsort(-burst)
        for i in order[:500]:
            s = (int(t[i]) - 3 * 60 * S) // (60 * S) * 60 * S
            day = datetime.fromtimestamp(s / S, timezone.utc).date().isoformat()
            if (day, s) in valid:
                out["cascade"] = (day, s)
                print(f"  cascade: ${burst[i] / 1e6:.1f}m liquidated within 60 s from "
                      f"{datetime.fromtimestamp(t[i] / S, timezone.utc):%Y-%m-%d %H:%M:%S} UTC")
                break
    else:
        print("  no liquidation files found for these dates: run python -m research.events first for the cascade window")
    return out


def run_window(name, day, w0, args, cache):
    w1 = w0 + WIN_MIN * 60 * S
    if day not in cache:
        cache.clear()
        cache[day] = load_signal(Namespace(intervals_dates=args.intervals_dates, model="ridge", method="adaptive",
                                           horizon=5, clean=args.clean, date=day))
    md_full, signal = cache[day]
    first = int(md_full.d_exch[md_full.g_start[0]])
    md = md_full.clip_hours((w1 + 6 * S - first) / 3.6e12)   # 6 s past the end so the last markouts settle
    L = LAT_MS * MS
    i0 = int(np.searchsorted(md.b_exch, w0, side="right")) - 1
    base = int(round(0.5 * (md.b_bid[max(i0, 0)] + md.b_ask[max(i0, 0)]) / TICK))

    out = {"name": name, "date": day, "start_utc": datetime.fromtimestamp(w0 / S, timezone.utc).isoformat(),
           "minutes": WIN_MIN, "tick": TICK, "base_ticks": base, "latency_ms": LAT_MS, "frame_ms": FRAME_MS,
           "strategies": {}}
    for key, (label, method, dial) in STRATS.items():
        t0 = time.time()
        if method == "conformal":
            strat = ConformalBudgetQuoter(signal, L, budget_bp=dial, lam0=lam0_for(day, method, dial))
        elif method == "conformal_q":
            strat = V3Quoter(signal, L, budget_bp=dial, lam0=lam0_for(day, method, dial), use_queue=True, use_liq=False)
        else:
            p = dict(size=0.01, max_pos=0.05, tick=0.1, theta=0.25, depth_k=5.0, rho_max=2.5)
            p.update(VARIANTS[method]); p["vol_max"] = dial
            strat = ForecastQuoter(signal, L, **p)
        lam, pre = [], [getattr(strat, "lam", None)]
        if hasattr(strat, "_settle"):
            orig = strat._settle

            def settle(t, mid, orig=orig, strat=strat):
                orig(t, mid)
                if t < w0:
                    pre[0] = strat.lam
                elif t < w1 and (not lam or lam[-1][1] != round(strat.lam, 4)):
                    lam.append([(t - w0) // MS, round(strat.lam, 4)])
            strat._settle = settle
        eng = RecEngine(md, strat, L, QUEUE, w0=w0, w1=w1, base=base)
        res = eng.run()
        fills = [f for f in res.fills if w0 <= f.t_ns < w1]
        ft = np.array([f.t_ns for f in fills], dtype=np.int64)
        m5 = mid_at(res, ft + 5 * S) if len(ft) else np.zeros(0)
        fl = []
        for f, m in zip(fills, m5):
            mk = f.side * (m - f.price) / f.price * 1e4 if np.isfinite(m) else None
            fl.append([(f.t_ns - w0) // MS, f.side, int(round(f.price / TICK)) - base, round(f.qty, 4),
                       round(mk, 3) if mk is not None else None, f.how, round(f.ahead0, 4)])
        rec = {"label": label, "events": eng.ev, "fills": fl}
        if hasattr(strat, "lam"):
            rec["lam"] = lam                       # threshold after each settled fill inside the window
            rec["lam_start"] = round(pre[0], 4)    # threshold on entering the window
            rec["budget"] = dial
        if key == "v3":
            out["frames"] = eng.frames
        out["strategies"][key] = rec
        mk = [x[4] for x in fl if x[4] is not None]
        print(f"    {key}: {len(fl)} fills in the window, mean 5 s markout "
              f"{(np.mean(mk) if mk else float('nan')):+.3f} bp ({time.time() - t0:.0f}s)")

    # market tape inside the window
    a, b = np.searchsorted(md.t_exch, [w0, w1])
    out["trades"] = [[int((md.t_exch[i] - w0) // MS), int(round(md.t_price[i] / TICK)) - base,
                      round(float(md.t_qty[i]), 4), int(md.t_side[i])] for i in range(a, b)]
    a, b = np.searchsorted(md.b_exch, [w0, w1])
    bb = []
    last = None
    for i in range(a, b):
        row = (int(round(md.b_bid[i] / TICK)) - base, int(round(md.b_ask[i] / TICK)) - base)
        if row != last:
            bb.append([int((md.b_exch[i] - w0) // MS), *row]); last = row
    out["bbo"] = bb
    # what the strategy knew: forecast, interval half-width, rho, every 100 ms (strategy's own view)
    sig = []
    for t in range(w0, w1, FRAME_MS * MS):
        x = signal.at(t - L)
        sig.append([(t - w0) // MS] + ([round(v, 4) for v in x] if x else [None, None, None]))
    out["signal"] = sig
    liq = liquidations(args.clean, [day])
    if liq is not None:
        t, sd, usd, ven = liq
        k = (t >= w0) & (t < w1)
        out["liquidations"] = [[int((tt - w0) // MS), int(s), round(float(u)), str(v)]
                               for tt, s, u, v in zip(t[k], sd[k], usd[k], ven[k])]
    else:
        out["liquidations"] = []
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--intervals-dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--window", action="append", default=[], help="NAME=YYYY-MM-DDTHH:MM (UTC)")
    args = ap.parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    wins = {}
    if args.auto:
        print("Choosing windows...")
        wins.update(pick_auto(args, dates))
    for w in args.window:
        name, ts = w.split("=")
        dt = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
        wins[name] = (dt.date().isoformat(), int(dt.timestamp()) * S)
    if not wins:
        raise SystemExit("nothing to export: use --auto or --window")
    outdir = Path("data/results/replay")
    outdir.mkdir(parents=True, exist_ok=True)
    index = []
    cache: dict = {}
    for name, (day, w0) in sorted(wins.items(), key=lambda kv: kv[1]):
        print(f"{name}: {datetime.fromtimestamp(w0 / S, timezone.utc):%Y-%m-%d %H:%M} UTC, {WIN_MIN} minutes")
        rec = run_window(name, day, w0, args, cache)
        p = outdir / f"replay_{name}.json"
        p.write_text(json.dumps(rec, separators=(",", ":")))
        index.append({"name": name, "file": p.name, "start_utc": rec["start_utc"], "mb": round(p.stat().st_size / 1e6, 2)})
        print(f"  saved {p} ({p.stat().st_size / 1e6:.1f} MB)")
    (outdir / "index.json").write_text(json.dumps(index, indent=1))
    print("Done. Send every file in data/results/replay.")


if __name__ == "__main__":
    main()
