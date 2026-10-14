"""Normalised output tables, written as Parquet partitioned by venue, symbol and UTC date.

Tables per symbol and day:
  deltas  the order book as a replayable stream. Rows sharing `msg` belong to
          one exchange message and must be applied together.
            kind 0  level update: set (side, price) to qty, qty 0 deletes
            kind 1  reset: clear the book; the level rows that follow with the
                    same msg rebuild it
            kind 2  invalid: the book cannot be trusted from here until the
                    next reset (gap, disconnect, stale feed)
  top     best N levels after every valid book message, for features
  trades  every trade, side +1 when the buyer was the aggressor, -1 when the seller was
  bbo     Binance bookTicker (best bid and offer, faster than the depth stream)

Every date file starts with a reset of the book carried over from the
previous day, so each day can be replayed on its own.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .book import ASK, BID, L2Book

TOP_N = 10
NS_PER_DAY = 86_400_000_000_000

KIND_LEVEL, KIND_RESET, KIND_INVALID = 0, 1, 2

SCHEMAS = {
    "deltas": pa.schema([
        ("recv_ns", pa.int64()), ("exch_ns", pa.int64()), ("msg", pa.int64()),
        ("kind", pa.int8()), ("side", pa.int8()), ("price", pa.float64()), ("qty", pa.float64()),
    ]),
    "top": pa.schema(
        [("recv_ns", pa.int64()), ("exch_ns", pa.int64())]
        + [(f"{s}_{f}_{i}", pa.float64()) for s in ("bid", "ask") for f in ("px", "qty") for i in range(1, TOP_N + 1)]
    ),
    "trades": pa.schema([
        ("recv_ns", pa.int64()), ("exch_ns", pa.int64()), ("trade_id", pa.string()),
        ("price", pa.float64()), ("qty", pa.float64()), ("side", pa.int8()),
    ]),
    "bbo": pa.schema([
        ("recv_ns", pa.int64()), ("exch_ns", pa.int64()),
        ("bid_px", pa.float64()), ("bid_qty", pa.float64()), ("ask_px", pa.float64()), ("ask_qty", pa.float64()),
    ]),
}


def day_of(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


class _Table:
    def __init__(self, path: Path, schema: pa.Schema, flush_rows: int) -> None:
        self.path, self.schema, self.flush_rows = path, schema, flush_rows
        self.cols: dict[str, list] = {n: [] for n in schema.names}
        self.writer: pq.ParquetWriter | None = None
        self.rows = 0

    def add(self, row: tuple) -> None:
        for name, v in zip(self.schema.names, row):
            self.cols[name].append(v)
        self.rows += 1
        if len(self.cols[self.schema.names[0]]) >= self.flush_rows:
            self.flush()

    def flush(self) -> None:
        if not self.cols[self.schema.names[0]]:
            return
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(pa.table(self.cols, schema=self.schema))
        self.cols = {n: [] for n in self.schema.names}

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


class SymbolOut:
    """All output for one venue and symbol, plus quality counters for the report."""

    def __init__(self, root: Path, venue: str, symbol: str, book: L2Book,
                 start_ns: int, end_ns: int, flush_rows: int = 250_000) -> None:
        self.root, self.venue, self.symbol, self.book = root, venue, symbol, book
        self.start_ns, self.end_ns, self.flush_rows = start_ns, end_ns, flush_rows
        self.first_ns: int | None = None  # first message inside the window, for coverage
        self.qty_mult = 1.0  # converts venue size units to coins (OKX swaps quote in contracts)
        self.tables: dict[str, _Table] = {}
        self.day: str | None = None
        self.msg = 0
        self.valid = False
        self._valid_since: int | None = None
        self.stats = {
            "book_msgs": 0, "resets": 0, "invalid_episodes": 0, "gaps": 0,
            "crossed": 0, "valid_ns": 0, "trades": 0, "bbo": 0,
        }

    # Time window and partitioning ------------------------------------------
    def _in_range(self, recv_ns: int) -> bool:
        if self.start_ns <= recv_ns < self.end_ns:
            if self.first_ns is None:
                self.first_ns = recv_ns
            return True
        return False

    def _table(self, name: str) -> _Table:
        t = self.tables.get(name)
        if t is None:
            path = self.root / self.venue / self.symbol / self.day / f"{name}.parquet"
            t = self.tables[name] = _Table(path, SCHEMAS[name], self.flush_rows)
        return t

    def _roll(self, recv_ns: int, exch_ns: int) -> None:
        """Start a new day file; open it with the current book state so it replays on its own."""
        day = day_of(recv_ns)
        if day == self.day:
            return
        for t in self.tables.values():
            t.close()
        self.tables = {}
        self.day = day
        if self.valid:
            self._write_reset(recv_ns, exch_ns)
        else:
            self.msg += 1
            self._table("deltas").add((recv_ns, exch_ns, self.msg, KIND_INVALID, 0, 0.0, 0.0))

    def _write_reset(self, recv_ns: int, exch_ns: int) -> None:
        self.msg += 1
        d = self._table("deltas")
        d.add((recv_ns, exch_ns, self.msg, KIND_RESET, 0, 0.0, 0.0))
        m = self.qty_mult
        for side, px, qty in self.book.levels():
            d.add((recv_ns, exch_ns, self.msg, KIND_LEVEL, side, px, qty * m))
        self._write_top(recv_ns, exch_ns)

    def _write_top(self, recv_ns: int, exch_ns: int) -> None:
        bids, asks = self.book.top(TOP_N)
        nan = float("nan")
        row = [recv_ns, exch_ns]
        for side in (bids, asks):
            pad = side + [(nan, nan)] * (TOP_N - len(side))
            row += [p for p, _ in pad] + [q * self.qty_mult for _, q in pad]
        self._table("top").add(tuple(row))

    def _set_valid(self, valid: bool, recv_ns: int) -> None:
        if valid and not self.valid:
            self._valid_since = recv_ns
        elif not valid and self.valid and self._valid_since is not None:
            self.stats["valid_ns"] += max(0, min(recv_ns, self.end_ns) - max(self._valid_since, self.start_ns))
        self.valid = valid

    # Called by venue processors ------------------------------------------------
    def reset(self, recv_ns: int, exch_ns: int) -> None:
        """The book has just been (re)built from a snapshot."""
        self._set_valid(True, recv_ns)
        self.stats["resets"] += 1
        if self._in_range(recv_ns):
            if day_of(recv_ns) != self.day:
                self._roll(recv_ns, exch_ns)  # a new day file opens with a reset of the current book
            else:
                self._write_reset(recv_ns, exch_ns)

    def invalidate(self, recv_ns: int, reason: str) -> None:
        if not self.valid:
            return
        self._set_valid(False, recv_ns)
        self.stats["invalid_episodes"] += 1
        if reason == "gap":
            self.stats["gaps"] += 1
        if self._in_range(recv_ns):
            self._roll(recv_ns, 0)
            self.msg += 1
            self._table("deltas").add((recv_ns, 0, self.msg, KIND_INVALID, 0, 0.0, 0.0))

    def begin_update(self, recv_ns: int, exch_ns: int) -> bool:
        """Call before applying an update message. Returns True if rows should be written."""
        if not self._in_range(recv_ns):
            return False
        self._roll(recv_ns, exch_ns)
        self.msg += 1
        return True

    def level(self, recv_ns: int, exch_ns: int, side: int, px: float, qty: float) -> None:
        self._table("deltas").add((recv_ns, exch_ns, self.msg, KIND_LEVEL, side, px, qty * self.qty_mult))

    def end_update(self, recv_ns: int, exch_ns: int, write: bool) -> None:
        self.stats["book_msgs"] += 1
        if self.book.crossed():
            self.stats["crossed"] += 1
        if write:
            self._write_top(recv_ns, exch_ns)

    def trade(self, recv_ns: int, exch_ns: int, trade_id: str, px: float, qty: float, side: int) -> None:
        if self._in_range(recv_ns):
            self._roll(recv_ns, exch_ns)
            self._table("trades").add((recv_ns, exch_ns, trade_id, px, qty * self.qty_mult, side))
            self.stats["trades"] += 1

    def bbo(self, recv_ns: int, exch_ns: int, bp: float, bq: float, ap: float, aq: float) -> None:
        if self._in_range(recv_ns):
            self._roll(recv_ns, exch_ns)
            self._table("bbo").add((recv_ns, exch_ns, bp, bq, ap, aq))
            self.stats["bbo"] += 1

    def close(self, last_ns: int) -> dict:
        self._set_valid(False, last_ns)
        for t in self.tables.values():
            t.close()
        span = max(1, min(last_ns, self.end_ns) - (self.first_ns or self.start_ns))
        s = dict(self.stats)
        s["valid_hours"] = round(s.pop("valid_ns") / 3.6e12, 3)
        s["data_hours"] = round(span / 3.6e12, 3)
        s["valid_share"] = round(s["valid_hours"] * 3.6e12 / span, 4)
        return s


__all__ = ["SymbolOut", "TOP_N", "BID", "ASK", "KIND_LEVEL", "KIND_RESET", "KIND_INVALID", "day_of"]
