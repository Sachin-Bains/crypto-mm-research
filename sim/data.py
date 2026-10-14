"""Market data for the simulator, loaded from the rebuilt Parquet tables into numpy arrays."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_EMPTY_I = np.zeros(0, dtype=np.int64)
_EMPTY_F = np.zeros(0, dtype=np.float64)


@dataclass
class MarketData:
    # book deltas, one row per level change; rows sharing a group belong to one exchange message
    d_exch: np.ndarray
    d_kind: np.ndarray
    d_side: np.ndarray
    d_price: np.ndarray
    d_qty: np.ndarray
    g_start: np.ndarray
    g_end: np.ndarray
    # trades
    t_exch: np.ndarray
    t_price: np.ndarray
    t_qty: np.ndarray
    t_side: np.ndarray   # aggressor: +1 buyer, -1 seller
    # real-time best bid and offer (Binance bookTicker), only rows where a best price changed
    b_exch: np.ndarray = field(default_factory=lambda: _EMPTY_I)
    b_bid: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    b_bid_qty: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    b_ask: np.ndarray = field(default_factory=lambda: _EMPTY_F)
    b_ask_qty: np.ndarray = field(default_factory=lambda: _EMPTY_F)

    @classmethod
    def from_arrays(cls, d_exch, d_msg, d_kind, d_side, d_price, d_qty, t_exch, t_price, t_qty, t_side,
                    bbo=None) -> "MarketData":
        d_msg = np.asarray(d_msg)
        n = len(d_msg)
        if n:
            starts = np.r_[0, np.flatnonzero(np.diff(d_msg)) + 1]
            ends = np.r_[starts[1:], n]
        else:
            starts = ends = np.zeros(0, dtype=np.int64)
        md = cls(
            np.asarray(d_exch, dtype=np.int64), np.asarray(d_kind, dtype=np.int8), np.asarray(d_side, dtype=np.int8),
            np.asarray(d_price, dtype=np.float64), np.asarray(d_qty, dtype=np.float64),
            starts.astype(np.int64), ends.astype(np.int64),
            np.asarray(t_exch, dtype=np.int64), np.asarray(t_price, dtype=np.float64),
            np.asarray(t_qty, dtype=np.float64), np.asarray(t_side, dtype=np.int8),
        )
        if bbo is not None:
            md.b_exch, md.b_bid, md.b_bid_qty, md.b_ask, md.b_ask_qty = (
                np.asarray(bbo[0], dtype=np.int64), *(np.asarray(x, dtype=np.float64) for x in bbo[1:]))
        return md

    @property
    def has_bbo(self) -> bool:
        return len(self.b_exch) > 0

    def clip_hours(self, hours: float) -> "MarketData":
        """Keep only the first `hours` of book data (and trades and BBO in that span), for quick runs."""
        if not len(self.g_start):
            return self
        t_end = self.d_exch[self.g_start[0]] + int(hours * 3.6e12)
        g_keep = int(np.searchsorted(self.d_exch[self.g_start], t_end, side="left"))
        n_rows = int(self.g_end[g_keep - 1]) if g_keep else 0
        n_tr = int(np.searchsorted(self.t_exch, t_end, side="left"))
        n_b = int(np.searchsorted(self.b_exch, t_end, side="left"))
        return MarketData(self.d_exch[:n_rows], self.d_kind[:n_rows], self.d_side[:n_rows], self.d_price[:n_rows],
                          self.d_qty[:n_rows], self.g_start[:g_keep], self.g_end[:g_keep],
                          self.t_exch[:n_tr], self.t_price[:n_tr], self.t_qty[:n_tr], self.t_side[:n_tr],
                          self.b_exch[:n_b], self.b_bid[:n_b], self.b_bid_qty[:n_b], self.b_ask[:n_b],
                          self.b_ask_qty[:n_b])


def load_day(clean_root: str | Path, venue: str, symbol: str, day: str, use_bbo: bool = True) -> MarketData:
    base = Path(clean_root) / venue / symbol / day
    d = pq.read_table(base / "deltas.parquet")
    t = pq.read_table(base / "trades.parquet")

    recv = d.column("recv_ns").to_numpy()
    exch = d.column("exch_ns").to_numpy().copy()
    # Invalid markers carry no exchange time; place them using the typical feed delay.
    has = exch > 0
    if has.any() and not has.all():
        delay = int(np.median(recv[has] - exch[has]))
        exch[~has] = recv[~has] - delay
    tt = t.column("exch_ns").to_numpy()
    order = np.argsort(tt, kind="stable")

    bbo = None
    if use_bbo and (base / "bbo.parquet").exists():
        b = pq.read_table(base / "bbo.parquet")
        be, bp, bq = b.column("exch_ns").to_numpy(), b.column("bid_px").to_numpy(), b.column("bid_qty").to_numpy()
        ap, aq = b.column("ask_px").to_numpy(), b.column("ask_qty").to_numpy()
        o = np.argsort(be, kind="stable")
        be, bp, bq, ap, aq = be[o], bp[o], bq[o], ap[o], aq[o]
        # Keep only rows where a best price changed: size-only updates do not move quotes or the mid.
        keep = np.ones(len(be), dtype=bool)
        keep[1:] = (bp[1:] != bp[:-1]) | (ap[1:] != ap[:-1])
        bbo = (be[keep], bp[keep], bq[keep], ap[keep], aq[keep])

    return MarketData.from_arrays(
        exch, d.column("msg").to_numpy(), d.column("kind").to_numpy(), d.column("side").to_numpy(),
        d.column("price").to_numpy(), d.column("qty").to_numpy(),
        tt[order], t.column("price").to_numpy()[order], t.column("qty").to_numpy()[order],
        t.column("side").to_numpy()[order], bbo=bbo,
    )
