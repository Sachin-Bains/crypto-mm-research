"""Tests for the forecasting pipeline.

1. No look-ahead: altering every message received after time T must not change
   any feature at or before T, nor any target whose horizon ends before T.
2. Signal recovery: a synthetic market where Binance follows OKX with a delay
   must show the cross-venue feature predicting Binance, out of sample.
3. Order flow imbalance matches a hand-worked example.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.evaluate import Ridge, metrics, walk_forward  # noqa: E402
from forecast.features import FEATURES, TARGETS, build_day  # noqa: E402
from rebuild.sink import SCHEMAS, TOP_N  # noqa: E402

DAY = "2026-10-14"
T0 = 1_792_022_400_000_000_000 + 3_600_000_000_000  # 01:00 UTC on DAY
S = 1_000_000_000


def write_market(root: Path, minutes: float, okx_lead_s: float = 0.0, seed: int = 0, mutate_after=None):
    """Synthetic clean tables for three venues. The common price is a random walk; Binance
    sees it okx_lead_s seconds after OKX. mutate_after: scramble everything received after it."""
    rng = np.random.default_rng(seed)
    n = int(minutes * 60 * 20)                      # 50 ms steps
    t = T0 + np.arange(n, dtype=np.int64) * 50_000_000
    walk = 80_000 + np.cumsum(rng.normal(0, 0.8, n))
    lag = int(okx_lead_s * 20)
    price = {"okx": walk, "bybit": walk, "binance": np.r_[np.full(lag, walk[0]), walk[: n - lag]] if lag else walk}
    for venue, sym in (("binance", "BTCUSDT"), ("bybit", "BTCUSDT"), ("okx", "BTC-USDT-SWAP")):
        base = root / venue / sym / DAY
        base.mkdir(parents=True, exist_ok=True)
        mid = np.round(price[venue], 1) + 0.05
        bq = rng.gamma(2, 1, n)
        aq = rng.gamma(2, 1, n)
        noise = rng.normal(0, 5, n)          # always drawn, so both markets use identical random numbers
        if mutate_after is not None:
            late = t > mutate_after
            mid = np.where(late, mid * 1.01 + noise, mid)
            bq = np.where(late, bq * 3, bq)
        bid, ask = mid - 0.05, mid + 0.05
        pq.write_table(pa.table({"recv_ns": t[:1], "exch_ns": t[:1], "msg": np.int64([1]), "kind": np.int8([1]),
                                 "side": np.int8([0]), "price": [0.0], "qty": [0.0]}, schema=SCHEMAS["deltas"]),
                       base / "deltas.parquet")
        top = {"recv_ns": t, "exch_ns": t}
        for side, px0, q0, sgn in (("bid", bid, bq, -1), ("ask", ask, aq, 1)):
            for k in range(1, TOP_N + 1):
                top[f"{side}_px_{k}"] = px0 + sgn * 0.1 * (k - 1)
        for side, q0 in (("bid", bq), ("ask", aq)):
            for k in range(1, TOP_N + 1):
                top[f"{side}_qty_{k}"] = q0 * (1 + 0.1 * k)
        pq.write_table(pa.table({c: top[c] for c in SCHEMAS["top"].names}, schema=SCHEMAS["top"]), base / "top.parquet")
        tr_t = t[::3]
        side = np.where(np.diff(np.r_[mid[0], mid])[::3] >= 0, 1, -1).astype(np.int8)
        tq = rng.exponential(0.3, len(tr_t))
        if mutate_after is not None:
            tq = np.where(tr_t > mutate_after, tq * 2, tq)
        pq.write_table(pa.table({"recv_ns": tr_t, "exch_ns": tr_t, "trade_id": [str(i) for i in range(len(tr_t))],
                                 "price": mid[::3], "qty": tq, "side": side},
                                schema=SCHEMAS["trades"]), base / "trades.parquet")
        if venue == "binance":
            pq.write_table(pa.table({"recv_ns": t, "exch_ns": t, "bid_px": bid, "bid_qty": bq,
                                     "ask_px": ask, "ask_qty": aq}, schema=SCHEMAS["bbo"]), base / "bbo.parquet")


def as_dict(tbl):
    return {c: tbl.column(c).to_numpy() for c in tbl.column_names}


def test_no_look_ahead(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    cut = T0 + 12 * 60 * S
    write_market(a, 20, seed=1)
    write_market(b, 20, seed=1, mutate_after=cut)
    fa, fb = as_dict(build_day(a, DAY)), as_dict(build_day(b, DAY))
    t = fa["recv_ns"]
    before = t <= cut
    assert before.sum() > 1000
    for name in FEATURES:
        np.testing.assert_array_equal(fa[name][before], fb[name][before], err_msg=name)
    for tg, h in zip(TARGETS, (1, 5, 30)):
        safe = t + h * S <= cut
        np.testing.assert_array_equal(fa[tg][safe], fb[tg][safe], err_msg=tg)
    # and the mutation really did change things after the cut
    assert not np.allclose(np.nan_to_num(fa["imb_l1"][~before]), np.nan_to_num(fb["imb_l1"][~before]))


def test_recovers_cross_venue_lead(tmp_path):
    write_market(tmp_path, 4 * 60, okx_lead_s=2.0, seed=2)   # 4 hours, Binance lags OKX by 2 s
    d = as_dict(build_day(tmp_path, DAY))
    ok = np.isfinite(d["okx_dev"]) & np.isfinite(d["y_5s"])
    assert np.corrcoef(d["okx_dev"][ok], d["y_5s"][ok])[0, 1] > 0.5
    preds = walk_forward(d, 3_600_000_000_000, {"ridge": Ridge})
    p, y = preds[("ridge", "y_5s")], d["y_5s"].astype(float)
    m = metrics(y[np.isfinite(p) & np.isfinite(y)], p[np.isfinite(p) & np.isfinite(y)])
    assert m["r2"] > 0.2 and m["ic"] > 0.5


def test_ofi_hand_example(tmp_path):
    """Three bbo events: bid size grows at the same price (+2), then the ask price ticks up
    (the old ask size 3 leaves: +3). OFI over the window = 5, scaled by current best size."""
    from forecast import features as F
    br = np.array([0, 10, 20], dtype=np.int64)
    bp, bq = np.array([100.0, 100.0, 100.0]), np.array([1.0, 3.0, 3.0])
    ap, aq = np.array([100.1, 100.1, 100.2]), np.array([3.0, 3.0, 4.0])
    e = np.zeros(3)
    e[1:] = ((bp[1:] >= bp[:-1]) * bq[1:] - (bp[1:] <= bp[:-1]) * bq[:-1]
             - (ap[1:] <= ap[:-1]) * aq[1:] + (ap[1:] >= ap[:-1]) * aq[:-1])
    assert list(e) == [0.0, 2.0, 3.0]
    assert F.window_sum(br, e, np.array([25]), 30)[0] == 5.0


def test_no_signal_no_skill(tmp_path):
    """Negative control: in a pure random walk with no lead, out-of-sample skill must be about zero."""
    write_market(tmp_path, 4 * 60, okx_lead_s=0.0, seed=3)
    d = as_dict(build_day(tmp_path, DAY))
    preds = walk_forward(d, 3_600_000_000_000, {"ridge": Ridge})
    for tg in TARGETS:
        p, y = preds[("ridge", tg)], d[tg].astype(float)
        ok = np.isfinite(p) & np.isfinite(y)
        assert metrics(y[ok], p[ok])["r2"] < 0.02, tg
