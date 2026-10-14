"""Per-venue rebuild: raw messages in, validated books and normalised trades out.

Each venue follows its own documented sequencing rules. A book is only
written while it can be trusted; anything else is marked invalid until the
next clean resync.
"""

from __future__ import annotations

import logging
import zlib
from collections import deque
from pathlib import Path

import orjson

from .book import ASK, BID, L2Book
from .sink import SymbolOut

log = logging.getLogger(__name__)
MS = 1_000_000
CHECK_LEVELS = 20


def same_top(a: L2Book, b: L2Book, n: int = CHECK_LEVELS) -> bool:
    return a.top(n) == b.top(n)


class _Venue:
    name = ""

    def __init__(self, out_root: Path, symbols: list[str], start_ns: int, end_ns: int) -> None:
        self.books = {s: L2Book() for s in symbols}
        self.out = {s: SymbolOut(out_root, self.name, s, self.books[s], start_ns, end_ns) for s in symbols}
        self.extra: dict[str, dict] = {s: {} for s in symbols}

    def bump(self, sym: str, key: str, n: int = 1) -> None:
        self.extra[sym][key] = self.extra[sym].get(key, 0) + n

    def handle(self, recv: int, tag: str, payload: str) -> None:
        raise NotImplementedError

    def finish(self, last_ns: int) -> dict:
        return {s: {**self.out[s].close(last_ns), **self.extra[s]} for s in self.out}


# --------------------------------------------------------------------------- Binance

class _BnState:
    def __init__(self) -> None:
        self.synced = False
        self.last_u: int | None = None
        self.buf: deque = deque(maxlen=50_000)  # recent depth events (recv, data), for syncing and checks
        self.snap: dict | None = None            # snapshot waiting to be used for sync
        self.check: dict | None = None           # snapshot waiting to be used as a cross-check


class BinanceRebuild(_Venue):
    """REST snapshot + diff depth, per Binance's documented procedure.

    Sync: drop events with u < lastUpdateId; the first applied event has
    U <= lastUpdateId <= u; then every event's pu must equal the previous u.

    Cross-check: every periodic REST snapshot taken while already in sync is
    replayed forward through the same events (snapshot + events from the one
    containing lastUpdateId) and compared with the live book's top 20 levels.
    Depth quantities are absolute, so the two must match exactly.
    """

    name = "binance"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.st = {s: _BnState() for s in self.books}

    def handle(self, recv: int, tag: str, payload: str) -> None:
        if tag == "ws_public":
            msg = orjson.loads(payload)
            d = msg.get("data")
            if not d:
                return
            stream = msg.get("stream", "")
            if "@depth" in stream:
                self._depth(recv, d)
            elif "@bookTicker" in stream and d["s"] in self.out:
                self.out[d["s"]].bbo(recv, d["T"] * MS, float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]))
        elif tag == "ws_market":
            msg = orjson.loads(payload)
            d = msg.get("data")
            if d and d.get("e") == "aggTrade" and d["s"] in self.out:
                self.out[d["s"]].trade(recv, d["T"] * MS, str(d["a"]), float(d["p"]), float(d["q"]),
                                       -1 if d["m"] else 1)
        elif tag.startswith("rest_depth:"):
            sym = tag.split(":", 1)[1]
            if sym in self.st:
                self._snapshot(recv, sym, orjson.loads(payload))
        elif tag == "event":
            ev = orjson.loads(payload)
            if ev.get("feed") == "ws_public" and ev.get("event") in ("open", "close", "stale"):
                for sym in self.st:
                    self._desync(sym, recv, "disconnect")

    def _desync(self, sym: str, recv: int, reason: str) -> None:
        st = self.st[sym]
        st.synced, st.last_u, st.check = False, None, None
        st.buf.clear()
        self.out[sym].invalidate(recv, reason)

    def _apply_raw(self, book: L2Book, d: dict) -> None:
        for px, q in d["b"]:
            book.set(BID, px, q)
        for px, q in d["a"]:
            book.set(ASK, px, q)

    def _depth(self, recv: int, d: dict) -> None:
        sym = d["s"]
        st = self.st.get(sym)
        if st is None:
            return
        if st.synced:
            if d["pu"] != st.last_u:
                self.bump(sym, "pu_breaks")
                self._desync(sym, recv, "gap")
                st.buf.append((recv, d))
                return
            st.buf.append((recv, d))
            out, book, exch = self.out[sym], self.books[sym], d["T"] * MS
            write = out.begin_update(recv, exch)
            for px, q in d["b"]:
                p, v = book.set(BID, px, q)
                if write:
                    out.level(recv, exch, BID, p, v)
            for px, q in d["a"]:
                p, v = book.set(ASK, px, q)
                if write:
                    out.level(recv, exch, ASK, p, v)
            st.last_u = d["u"]
            out.end_update(recv, exch, write)
            if st.check is not None:
                self._run_check(sym)
        else:
            st.buf.append((recv, d))
            if st.snap is not None:
                self._try_sync(sym, recv)

    def _snapshot(self, recv: int, sym: str, body: dict) -> None:
        st = self.st[sym]
        snap = {"L": body["lastUpdateId"], "bids": body["bids"], "asks": body["asks"]}
        if st.synced:
            st.check = snap
            self._run_check(sym)
        else:
            st.snap = snap
            self._try_sync(sym, recv)

    def _replay_from(self, st: _BnState, snap: dict) -> tuple[L2Book, int, int] | None:
        """Snapshot + buffered events from the one containing L. None if L is not in the buffer."""
        L = snap["L"]
        events = list(st.buf)
        k = next((i for i, (_, d) in enumerate(events) if d["U"] <= L <= d["u"]), None)
        if k is None:
            return None
        book = L2Book()
        book.load(snap["bids"], snap["asks"])
        prev_u = None
        for _, d in events[k:]:
            if prev_u is not None and d["pu"] != prev_u:
                return None
            self._apply_raw(book, d)
            prev_u = d["u"]
        return book, prev_u, events[-1][1]["T"] * MS

    def _try_sync(self, sym: str, recv: int) -> None:
        st = self.st[sym]
        L = st.snap["L"]
        while st.buf and st.buf[0][1]["u"] < L:
            st.buf.popleft()
        if not st.buf:
            return
        if st.buf[0][1]["U"] > L:
            # Snapshot is older than the stream we hold: it can never be bridged.
            self.bump(sym, "stale_snapshots")
            st.snap = None
            return
        res = self._replay_from(st, st.snap)
        if res is None:
            return
        book, last_u, exch = res
        self.books[sym].bids, self.books[sym].asks = book.bids, book.asks
        st.synced, st.last_u, st.snap = True, last_u, None
        self.out[sym].reset(recv, exch)

    def _run_check(self, sym: str) -> None:
        st = self.st[sym]
        snap = st.check
        if st.buf and st.buf[0][1]["U"] > snap["L"]:
            st.check = None  # snapshot predates the buffer, cannot check
            self.bump(sym, "checks_skipped")
            return
        res = self._replay_from(st, snap)
        if res is None:
            return  # the event containing L has not arrived yet
        st.check = None
        self.bump(sym, "snapshot_checks")
        if same_top(res[0], self.books[sym]):
            self.bump(sym, "snapshot_checks_matched")
        else:
            self.bump(sym, "snapshot_checks_failed")


# --------------------------------------------------------------------------- Bybit

class BybitRebuild(_Venue):
    """Snapshot on subscribe, then deltas whose update id u advances by exactly one.

    The +1 rule is not in Bybit's docs; it was measured on live data
    (1803 of 1803 deltas). Any break is treated as a gap until the next snapshot.
    """

    name = "bybit"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.synced = {s: False for s in self.books}
        self.last_u: dict[str, int] = {}

    def handle(self, recv: int, tag: str, payload: str) -> None:
        if tag == "ws":
            if '"topic":"orderbook.' in payload[:80]:
                self._book(recv, orjson.loads(payload))
            elif '"topic":"publicTrade.' in payload[:80]:
                msg = orjson.loads(payload)
                for t in msg["data"]:
                    if t["s"] in self.out:
                        self.out[t["s"]].trade(recv, t["T"] * MS, t["i"], float(t["p"]), float(t["v"]),
                                               1 if t["S"] == "Buy" else -1)
        elif tag == "event":
            ev = orjson.loads(payload)
            if ev.get("event") in ("open", "close", "stale"):
                for s in self.books:
                    self.synced[s] = False
                    self.out[s].invalidate(recv, "disconnect")

    def _book(self, recv: int, msg: dict) -> None:
        sym = msg["topic"].rsplit(".", 1)[1]
        if sym not in self.books:
            return
        d = msg["data"]
        exch = int(msg.get("cts") or msg["ts"]) * MS
        u = d["u"]
        book, out = self.books[sym], self.out[sym]
        if msg["type"] == "snapshot":
            if self.synced[sym] and u == self.last_u.get(sym):
                # Resubscribe snapshot at the same update id: an exact cross-check.
                ref = L2Book()
                ref.load(d["b"], d["a"])
                self.bump(sym, "snapshot_checks")
                self.bump(sym, "snapshot_checks_matched" if same_top(ref, book) else "snapshot_checks_failed")
            book.load(d["b"], d["a"])
            self.synced[sym], self.last_u[sym] = True, u
            out.reset(recv, exch)
            return
        if not self.synced[sym]:
            return
        if u != self.last_u[sym] + 1:
            self.bump(sym, "u_breaks")
            self.synced[sym] = False
            out.invalidate(recv, "gap")
            return
        write = out.begin_update(recv, exch)
        for px, q in d["b"]:
            p, v = book.set(BID, px, q)
            if write:
                out.level(recv, exch, BID, p, v)
        for px, q in d["a"]:
            p, v = book.set(ASK, px, q)
            if write:
                out.level(recv, exch, ASK, p, v)
        self.last_u[sym] = u
        out.end_update(recv, exch, write)


# --------------------------------------------------------------------------- OKX

OKX_DEFAULT_CTVAL = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.1}


def okx_checksum(book: L2Book) -> int:
    """CRC32 over the top 25 levels, bids and asks interleaved as price:size, as a signed int."""
    bids, asks = book.top_strings(25)
    parts: list[str] = []
    for i in range(25):
        if i < len(bids):
            parts += bids[i]
        if i < len(asks):
            parts += asks[i]
    crc = zlib.crc32(":".join(parts).encode())
    return crc - (1 << 32) if crc >= (1 << 31) else crc


class OkxRebuild(_Venue):
    """Snapshot on subscribe, then updates where prevSeqId must equal the previous seqId.

    The exchange checksum is verified after every message and reported as a
    match rate. It does not invalidate the book: the rule is implemented from
    the documented description and is confirmed or refuted by that rate.
    """

    name = "okx"

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.synced = {s: False for s in self.books}
        self.last_seq: dict[str, int] = {}
        for s, o in self.out.items():
            o.qty_mult = OKX_DEFAULT_CTVAL.get(s, 1.0)

    def handle(self, recv: int, tag: str, payload: str) -> None:
        if tag == "ws":
            if '"channel":"books' in payload[:80]:
                msg = orjson.loads(payload)
                if "data" in msg:
                    self._book(recv, msg)
            elif '"channel":"trades"' in payload[:80]:
                msg = orjson.loads(payload)
                for t in msg.get("data", []):
                    if t["instId"] in self.out:
                        self.out[t["instId"]].trade(recv, int(t["ts"]) * MS, t["tradeId"], float(t["px"]),
                                                    float(t["sz"]), 1 if t["side"] == "buy" else -1)
        elif tag == "meta":
            for item in orjson.loads(payload).get("data", []):
                inst, ct = item.get("instId"), item.get("ctVal")
                if inst in self.out and ct:
                    if float(ct) != self.out[inst].qty_mult:
                        log.warning("okx %s contract value %s differs from default", inst, ct)
                    self.out[inst].qty_mult = float(ct)
        elif tag == "event":
            ev = orjson.loads(payload)
            if ev.get("event") in ("open", "close", "stale"):
                for s in self.books:
                    self.synced[s] = False
                    self.out[s].invalidate(recv, "disconnect")

    def _verify(self, sym: str, expected) -> None:
        if expected is None:
            return
        self.bump(sym, "checksums")
        if okx_checksum(self.books[sym]) == int(expected):
            self.bump(sym, "checksums_matched")

    def _book(self, recv: int, msg: dict) -> None:
        sym = msg["arg"]["instId"]
        if sym not in self.books:
            return
        book, out = self.books[sym], self.out[sym]
        for d in msg["data"]:
            exch = int(d["ts"]) * MS
            seq, prev = d["seqId"], d["prevSeqId"]
            if msg.get("action") == "snapshot":
                book.load(d["bids"], d["asks"])
                self.synced[sym], self.last_seq[sym] = True, seq
                out.reset(recv, exch)
                self._verify(sym, d.get("checksum"))
                continue
            if not self.synced[sym]:
                continue
            if prev != self.last_seq[sym]:
                self.bump(sym, "seq_breaks")
                self.synced[sym] = False
                out.invalidate(recv, "gap")
                continue
            write = out.begin_update(recv, exch)
            for lvl in d["bids"]:
                p, v = book.set(BID, lvl[0], lvl[1])
                if write:
                    out.level(recv, exch, BID, p, v)
            for lvl in d["asks"]:
                p, v = book.set(ASK, lvl[0], lvl[1])
                if write:
                    out.level(recv, exch, ASK, p, v)
            self.last_seq[sym] = seq
            out.end_update(recv, exch, write)
            self._verify(sym, d.get("checksum"))


VENUES = {"binance": BinanceRebuild, "bybit": BybitRebuild, "okx": OkxRebuild}
