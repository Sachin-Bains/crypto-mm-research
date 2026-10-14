"""Feature and target dataset for short-horizon forecasting of the Binance BTC mid.

One row every 100 ms. Everything is on the receive clock: a row at time t uses
only messages our machine had received by t, and the targets are how the
Binance mid we received moved over the next 1, 5 and 30 seconds. Using the
exchanges' own timestamps instead would let a cross-venue feature "see" a move
before we could have known about it.

Features (all known at t):
  imb_l1            (bid size - ask size) / (bid size + ask size) at the best prices
  micro_bps         microprice minus mid: the size-weighted fair price, in bps
  imb_l5            depth imbalance over the best 5 levels per side
  ofi_1s, ofi_5s    order flow imbalance (Cont, Kukanov and Stoikov 2014) at the best
                    prices over the window, scaled by current best-level size
  tfi_1s, tfi_5s    signed trade volume / total trade volume over the window
  ntrades_5s        log(1 + number of trades in the last 5 s)
  ret_1s, ret_5s    past change in the Binance mid, bps
  spread_bps        best ask minus best bid, bps of mid
  rv_30s            realised volatility of 100 ms mid changes over 30 s, bps
  {bybit,okx}_dev   that venue's mid minus Binance's, bps, relative to its 5-minute average
  {bybit,okx}_lead  that venue's last-1 s move minus Binance's last-1 s move, bps
  {bybit,okx}_imb   that venue's best-level imbalance
Targets:
  y_1s, y_5s, y_30s change in the Binance mid over the next h seconds, bps
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

STEP_NS = 100_000_000
WARMUP_NS = 300 * 1_000_000_000          # 5 minutes, for the rolling basis average
HORIZONS_S = (1, 5, 30)
STALE_NS = 5 * 1_000_000_000             # a feed silent this long is treated as missing
OTHER_VENUES = {"bybit": "BTCUSDT", "okx": "BTC-USDT-SWAP"}

FEATURES = [
    "imb_l1", "micro_bps", "imb_l5", "ofi_1s", "ofi_5s", "tfi_1s", "tfi_5s", "ntrades_5s",
    "ret_1s", "ret_5s", "spread_bps", "rv_30s",
    "bybit_dev", "bybit_lead", "bybit_imb", "okx_dev", "okx_lead", "okx_imb",
]
TARGETS = [f"y_{h}s" for h in HORIZONS_S]


# --------------------------------------------------------------------------- helpers

def asof(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the last event at or before each grid time (-1 if none)."""
    return np.searchsorted(times, grid, side="right") - 1


def take(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    out = np.full(len(idx), np.nan)
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def window_sum(times: np.ndarray, values: np.ndarray, grid: np.ndarray, width_ns: int) -> np.ndarray:
    """Sum of values for events in (t - width, t]."""
    c = np.r_[0.0, np.cumsum(values)]
    hi = np.searchsorted(times, grid, side="right")
    lo = np.searchsorted(times, grid - width_ns, side="right")
    return c[hi] - c[lo]


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over the last n grid points, ignoring NaNs (needs at least n/2 values)."""
    ok = ~np.isnan(x)
    cs = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    cn = np.r_[0, np.cumsum(ok)]
    i = np.arange(1, len(x) + 1)
    j = np.maximum(i - n, 0)
    cnt = cn[i] - cn[j]
    out = (cs[i] - cs[j]) / np.maximum(cnt, 1)
    out[cnt < n // 2] = np.nan
    return out


def validity(base: Path, grid: np.ndarray) -> np.ndarray:
    """True where the venue's rebuilt book was in sync (last reset more recent than last invalid)."""
    path = base / "deltas.parquet"
    if not path.exists():
        return np.zeros(len(grid), dtype=bool)
    t = pq.read_table(path, columns=["recv_ns", "kind"])
    t = t.filter(pc.not_equal(t.column("kind"), 0))
    recv, kind = t.column("recv_ns").to_numpy(), t.column("kind").to_numpy()
    idx = asof(recv, grid)
    ok = idx >= 0
    out = np.zeros(len(grid), dtype=bool)
    out[ok] = kind[idx[ok]] == 1
    return out


def read(path: Path, cols: list[str]) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    t = pq.read_table(path, columns=cols)
    return {c: t.column(c).to_numpy() for c in cols}


def fresh(times: np.ndarray, idx: np.ndarray, grid: np.ndarray) -> np.ndarray:
    ok = idx >= 0
    out = np.zeros(len(grid), dtype=bool)
    out[ok] = grid[ok] - times[idx[ok]] <= STALE_NS
    return out


# --------------------------------------------------------------------------- build

def build_day(clean: Path, day: str) -> pa.Table:
    bbase = clean / "binance" / "BTCUSDT" / day
    bbo = read(bbase / "bbo.parquet", ["recv_ns", "bid_px", "bid_qty", "ask_px", "ask_qty"])
    if bbo is None or not len(bbo["recv_ns"]):
        raise SystemExit(f"No Binance bbo data for {day}; run the rebuild first")
    br, bp, bq, ap, aq = bbo["recv_ns"], bbo["bid_px"], bbo["bid_qty"], bbo["ask_px"], bbo["ask_qty"]

    first = (int(br[0]) // STEP_NS + 1) * STEP_NS + WARMUP_NS
    last = int(br[-1]) - max(HORIZONS_S) * 1_000_000_000
    if last <= first:
        raise SystemExit(f"{day}: not enough data for a feature window")
    grid = np.arange(first, last, STEP_NS, dtype=np.int64)
    f: dict[str, np.ndarray] = {}

    # Binance best bid and offer -------------------------------------------------
    i = asof(br, grid)
    b_ok = fresh(br, i, grid) & validity(bbase, grid)
    bid, ask, bqs, aqs = take(bp, i), take(ap, i), take(bq, i), take(aq, i)
    mid = 0.5 * (bid + ask)
    f["imb_l1"] = (bqs - aqs) / (bqs + aqs)
    micro = (ask * bqs + bid * aqs) / (bqs + aqs)
    f["micro_bps"] = (micro - mid) / mid * 1e4
    f["spread_bps"] = (ask - bid) / mid * 1e4

    # Order flow imbalance at the best prices, per bbo event
    e = np.zeros(len(br))
    e[1:] = ((bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1]
             - (ap[1:] <= ap[:-1]) * aq[1:] + (ap[1:] >= ap[:-1]) * aq[:-1])
    depth = bqs + aqs
    for w in (1, 5):
        f[f"ofi_{w}s"] = window_sum(br, e, grid, w * 1_000_000_000) / depth

    # Depth imbalance over 5 levels (100 ms depth stream)
    top = read(bbase / "top.parquet", ["recv_ns"] + [f"{s}_qty_{k}" for s in ("bid", "ask") for k in range(1, 6)])
    j = asof(top["recv_ns"], grid)
    sb = sum(np.nan_to_num(take(top[f"bid_qty_{k}"], j)) for k in range(1, 6))
    sa = sum(np.nan_to_num(take(top[f"ask_qty_{k}"], j)) for k in range(1, 6))
    f["imb_l5"] = np.where(sb + sa > 0, (sb - sa) / np.maximum(sb + sa, 1e-12), np.nan)
    f["imb_l5"][~fresh(top["recv_ns"], j, grid)] = np.nan

    # Trades
    tr = read(bbase / "trades.parquet", ["recv_ns", "qty", "side"])
    tt, tq, ts = tr["recv_ns"], tr["qty"], tr["side"].astype(np.float64)
    for w in (1, 5):
        signed = window_sum(tt, tq * ts, grid, w * 1_000_000_000)
        total = window_sum(tt, tq, grid, w * 1_000_000_000)
        f[f"tfi_{w}s"] = np.where(total > 0, signed / np.maximum(total, 1e-12), 0.0)
    f["ntrades_5s"] = np.log1p(window_sum(tt, np.ones(len(tt)), grid, 5_000_000_000))

    # Past moves and volatility
    def mid_at(t):
        k = asof(br, t)
        return 0.5 * (take(bp, k) + take(ap, k))

    for w in (1, 5):
        prev = mid_at(grid - w * 1_000_000_000)
        f[f"ret_{w}s"] = (mid - prev) / prev * 1e4
    step_ret = np.r_[np.nan, np.diff(mid) / mid[:-1] * 1e4]
    sq = np.where(np.isnan(step_ret), 0.0, step_ret ** 2)
    cs = np.r_[0.0, np.cumsum(sq)]
    n = 300
    idx = np.arange(1, len(grid) + 1)
    f["rv_30s"] = np.sqrt(cs[idx] - cs[np.maximum(idx - n, 0)])

    # Other venues -------------------------------------------------------------------
    for venue, sym in OTHER_VENUES.items():
        vbase = clean / venue / sym / day
        vt = read(vbase / "top.parquet", ["recv_ns", "bid_px_1", "ask_px_1", "bid_qty_1", "ask_qty_1"])
        if vt is None:
            for k in ("dev", "lead", "imb"):
                f[f"{venue}_{k}"] = np.full(len(grid), np.nan)
            continue
        k = asof(vt["recv_ns"], grid)
        v_ok = fresh(vt["recv_ns"], k, grid) & validity(vbase, grid)
        vmid = 0.5 * (take(vt["bid_px_1"], k) + take(vt["ask_px_1"], k))
        vb, va = take(vt["bid_qty_1"], k), take(vt["ask_qty_1"], k)
        dev = (vmid - mid) / mid * 1e4
        dev[~v_ok] = np.nan
        f[f"{venue}_dev"] = dev - rolling_mean(dev, int(WARMUP_NS // STEP_NS))
        k1 = asof(vt["recv_ns"], grid - 1_000_000_000)
        vprev = 0.5 * (take(vt["bid_px_1"], k1) + take(vt["ask_px_1"], k1))
        lead = (vmid - vprev) / vprev * 1e4 - f["ret_1s"]
        lead[~v_ok] = np.nan
        f[f"{venue}_lead"] = lead
        imb = (vb - va) / (vb + va)
        imb[~v_ok] = np.nan
        f[f"{venue}_imb"] = imb

    # Targets ------------------------------------------------------------------------------
    y = {}
    for h in HORIZONS_S:
        th = grid + h * 1_000_000_000
        fut = mid_at(th)
        ok = b_ok & validity(bbase, th)
        yy = (fut - mid) / mid * 1e4
        yy[~ok] = np.nan
        y[f"y_{h}s"] = yy

    for name in FEATURES:
        f[name] = np.where(b_ok, f[name], np.nan)

    cols = {"recv_ns": grid}
    cols.update({k: f[k].astype(np.float32) for k in FEATURES})
    cols.update({k: y[k].astype(np.float32) for k in TARGETS})
    cols["mid"] = mid
    return pa.table(cols)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the 100 ms feature/target dataset for one or more days")
    ap.add_argument("--dates", required=True, help="comma-separated UTC dates, e.g. 2026-10-14,2026-10-15")
    ap.add_argument("--clean", default="data/clean")
    ap.add_argument("--out", default="data/features")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for day in [d.strip() for d in args.dates.split(",") if d.strip()]:
        t0 = time.time()
        tbl = build_day(Path(args.clean), day)
        pq.write_table(tbl, out / f"binance_BTCUSDT_{day}.parquet", compression="zstd")
        usable = np.isfinite(tbl.column("y_1s").to_numpy()).mean()
        print(f"{day}: {tbl.num_rows:,} rows ({tbl.num_rows / 36000:.1f} h), "
              f"{usable:.1%} with a valid target, {time.time() - t0:.0f}s")
        cov = {k: float(np.isfinite(tbl.column(k).to_numpy()).mean()) for k in FEATURES}
        low = {k: f"{v:.0%}" for k, v in cov.items() if v < 0.9}
        if low:
            print(f"  features with under 90% coverage: {low}")


if __name__ == "__main__":
    main()
