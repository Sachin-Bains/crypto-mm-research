"""Which venue moves first: lead-lag between Binance, Bybit and OKX BTC perpetuals.

Usage:
  python -m research.leadlag --dates 2026-10-15,2026-10-16,2026-10-17,2026-10-18

Method. Each venue's mid (Binance from the real-time best bid and offer, Bybit and OKX from
their book updates) is sampled every 10 ms on its last value (previous-tick), turned into log
returns, and cross-correlated hour by hour at lags up to +-2 s using FFTs:
    c(l) = sum_k rA[k] rB[k + l] / sqrt(sum rA^2 sum rB^2)
A peak at positive l means A's moves are followed by B's l later: A leads. The lead-lag ratio
(Huth and Abergel) LLR = sum_{l>0} c(l)^2 / sum_{l<0} c(l)^2 summarises it: above 1, A leads.
Previous-tick sampling shrinks correlations at fine scales (the Epps effect); it does not move
where the peak sits, which is what is used here.

Two clocks, two questions:
  exchange  timestamps set by each exchange: who moves first at the source
  receive   when our machine in Glasgow got each update: whose data reaches us first
Exchange clocks are compared through the per-minute clock samples (offset = server time minus
the midpoint of our request), so the report states how far apart they are before any lead is read.

Writes data/results/research_leadlag_<first>_<last>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

S = 1_000_000_000
STEP_NS = 10_000_000
MAXLAG = 200                         # 200 x 10 ms = 2 s
BOOK = {"binance": "BTCUSDT", "bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP"}
PAIRS = (("binance", "bybit"), ("binance", "okx"), ("bybit", "okx"))


def load_mid(clean: Path, venue: str, day: str, clock: str):
    base = clean / venue / BOOK[venue] / day
    tcol = "exch_ns" if clock == "exchange" else "recv_ns"
    if venue == "binance" and (base / "bbo.parquet").exists():
        t = pq.read_table(base / "bbo.parquet", columns=[tcol, "bid_px", "ask_px"])
        b, a = t.column("bid_px").to_numpy(), t.column("ask_px").to_numpy()
    else:
        t = pq.read_table(base / "top.parquet", columns=[tcol, "bid_px_1", "ask_px_1"])
        b, a = t.column("bid_px_1").to_numpy(), t.column("ask_px_1").to_numpy()
    x = t.column(tcol).to_numpy().astype(np.int64)
    m = 0.5 * (b + a)
    ok = np.isfinite(m) & (m > 0) & (x > 0)
    x, m = x[ok], m[ok]
    o = np.argsort(x, kind="stable")
    return x[o], m[o]


def hour_returns(x, m, h0):
    lo, hi = np.searchsorted(x, [h0, h0 + 3600 * S])
    if hi - lo < 2000:
        return None
    grid = h0 + np.arange(0, 3600 * S, STEP_NS, dtype=np.int64)
    i = np.searchsorted(x, grid, side="right") - 1
    if i[0] < 0:
        return None
    return np.diff(np.log(m[i]))


def xcorr(a, b):
    n = len(a)
    size = 1 << int(np.ceil(np.log2(2 * n)))
    fa, fb = np.fft.rfft(a, size), np.fft.rfft(b, size)
    full = np.fft.irfft(np.conj(fa) * fb, size)          # full[l] = sum_k a[k] b[k + l]
    lags = np.arange(-MAXLAG, MAXLAG + 1)
    c = np.array([full[l % size] for l in lags])
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return c / den if den > 0 else np.full(len(lags), np.nan)


def llr(c):
    k = len(c) // 2
    pos, neg = np.nansum(c[k + 1:] ** 2), np.nansum(c[:k] ** 2)
    return float(pos / neg) if neg > 0 else float("nan")


def clock_offsets(clean: Path, days):
    rows = {}
    for d in days:
        f = clean / "events" / d / "clock.parquet"
        if not f.exists():
            continue
        t = pq.read_table(f)
        v, t0, t1, sv = (t.column(c).to_numpy(zero_copy_only=False) for c in ("venue", "t0_ns", "t1_ns", "server_ns"))
        for name in set(v.tolist()):
            k = (v == name) & (sv > 0)
            off = (sv[k] - (t0[k] + t1[k]) / 2) / 1e6
            rtt = (t1[k] - t0[k]) / 1e6
            rows.setdefault(name, {"off": [], "rtt": []})
            rows[name]["off"] += off.tolist()
            rows[name]["rtt"] += rtt.tolist()
    out = {}
    for name, r in rows.items():
        off, rtt = np.array(r["off"]), np.array(r["rtt"])
        if not len(off):
            continue
        out[name] = {"n": int(len(off)), "offset_ms_median": float(np.median(off)), "offset_ms_iqr": [float(x) for x in np.percentile(off, [25, 75])],
                     "rtt_ms_median": float(np.median(rtt)), "rtt_ms_p10": float(np.percentile(rtt, 10))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--clean", default="data/clean")
    args = ap.parse_args()
    days = [d.strip() for d in args.dates.split(",") if d.strip()]
    clean = Path(args.clean)
    lags_s = (np.arange(-MAXLAG, MAXLAG + 1) * STEP_NS / S).round(3)
    report = {"dates": days, "lags_s": lags_s.tolist(), "clocks": clock_offsets(clean, days), "pairs": {}}
    for clock in ("exchange", "receive"):
        per = {p: {"curves": [], "llr": [], "vol": []} for p in PAIRS}
        for day in days:
            mids = {v: load_mid(clean, v, day, clock) for v in BOOK}
            x0 = mids["binance"][0]
            d0 = int(x0[0] // (86400 * S) * 86400 * S) if len(x0) else int(np.datetime64(day, "ns").astype(np.int64))
            for h in range(24):
                h0 = d0 + h * 3600 * S
                r = {v: hour_returns(*mids[v], h0) for v in BOOK}
                if r["binance"] is None:
                    continue
                vol = float(np.sqrt(np.sum(r["binance"] ** 2)) * 1e4)
                for a, b in PAIRS:
                    if r[a] is None or r[b] is None:
                        continue
                    c = xcorr(r[a], r[b])
                    per[(a, b)]["curves"].append(c)
                    per[(a, b)]["llr"].append(llr(c))
                    per[(a, b)]["vol"].append(vol)
            print(f"{clock} clock, {day} done")
        for (a, b), d in per.items():
            if not d["curves"]:
                continue
            C, R, V = np.vstack(d["curves"]), np.array(d["llr"]), np.array(d["vol"])
            mean = np.nanmean(C, axis=0)
            q1, q2 = np.quantile(V, [1 / 3, 2 / 3])
            by_vol = {}
            for name, m in (("calm", V <= q1), ("normal", (V > q1) & (V <= q2)), ("volatile", V > q2)):
                cm = np.nanmean(C[m], axis=0)
                by_vol[name] = {"hours": int(m.sum()), "peak_lag_s": float(lags_s[int(np.nanargmax(cm))]),
                                "llr_median": float(np.nanmedian(R[m]))}
            report["pairs"].setdefault(clock, {})[f"{a}->{b}"] = {
                "hours": int(len(R)), "peak_lag_s": float(lags_s[int(np.nanargmax(mean))]), "peak_corr": float(np.nanmax(mean)),
                "corr_at_0": float(mean[MAXLAG]), "llr_median": float(np.nanmedian(R)), "share_hours_a_leads": float(np.mean(R > 1)),
                "by_volatility": by_vol, "curve": [round(float(x), 5) for x in mean]}

    tag = f"{days[0]}_{days[-1]}"
    out = Path("data/results") / f"research_leadlag_{tag}.json"
    out.write_text(json.dumps(report))
    print("\nClock offsets (exchange server time minus our time at the request midpoint) and round trips from here:")
    for v, c in sorted(report["clocks"].items()):
        print(f"  {v:<8} offset median {c['offset_ms_median']:+.1f} ms (IQR {c['offset_ms_iqr'][0]:+.1f} to {c['offset_ms_iqr'][1]:+.1f}), "
              f"round trip median {c['rtt_ms_median']:.0f} ms (fastest 10% {c['rtt_ms_p10']:.0f} ms), {c['n']:,} samples")
    for clock, pairs in report["pairs"].items():
        print(f"\n{clock} clock: A->B, positive lag = A moves first")
        print(f"  {'pair':<18}{'hours':>6}{'peak lag':>10}{'corr @peak':>12}{'corr @0':>10}{'LLR':>7}{'A leads':>9}   peak lag calm / normal / volatile")
        for k, p in pairs.items():
            bv = p["by_volatility"]
            print(f"  {k:<18}{p['hours']:>6}{p['peak_lag_s']:>+9.2f}s{p['peak_corr']:>12.3f}{p['corr_at_0']:>10.3f}{p['llr_median']:>7.2f}"
                  f"{p['share_hours_a_leads']:>9.0%}   {bv['calm']['peak_lag_s']:+.2f} / {bv['normal']['peak_lag_s']:+.2f} / {bv['volatile']['peak_lag_s']:+.2f} s")
    edge = any(abs(p["peak_lag_s"]) >= MAXLAG * STEP_NS / S for pr in report["pairs"].values() for p in pr.values())
    if edge:
        print("\nA peak at +-2.00 s sits on the edge of the window: the true lag is at least that long, or there is no clear peak.")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
