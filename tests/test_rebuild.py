"""End-to-end rebuild tests on synthetic exchange data with a known true book.

A random order book evolves; each venue's messages are generated from it in
that venue's exact wire format, including sequencing fields, late REST
snapshots, a dropped message and a disconnect. The rebuilt output is then
replayed and compared against the true book after every message.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import orjson
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuild.__main__ import run  # noqa: E402
from rebuild.book import ASK, BID, L2Book  # noqa: E402
from rebuild.venues import okx_checksum  # noqa: E402
from recorder.storage import RawWriter  # noqa: E402

DAY = datetime(2026, 10, 14, tzinfo=timezone.utc)
T0 = int((DAY + timedelta(hours=23, minutes=59, seconds=30)).timestamp() * 1e9)  # crosses midnight
STEP = 100_000_000  # 100 ms between messages
CFG = {"symbols": {"binance": ["BTCUSDT"], "bybit": ["BTCUSDT"], "okx": ["BTC-USDT-SWAP"]}}


class TrueBook:
    """Random-walk L2 book with prices on a 0.1 tick."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.bids = {round(100 - 0.1 * i, 1): self._q() for i in range(1, 40)}
        self.asks = {round(100 + 0.1 * i, 1): self._q() for i in range(0, 40)}

    def _q(self) -> float:
        return round(self.rng.uniform(0.001, 5), 3)

    def step(self):
        """Mutate a few levels; return absolute updates as string pairs per side."""
        b, a = [], []
        for _ in range(self.rng.randint(1, 6)):
            side = self.rng.choice((BID, ASK))
            levels, out = (self.bids, b) if side == BID else (self.asks, a)
            px = self.rng.choice(sorted(levels))
            if self.rng.random() < 0.3:
                levels.pop(px)
                qty = 0.0
            else:
                qty = levels[px] = self._q()
            out.append([f"{px:.1f}", f"{qty:.3f}"])
        # keep the book uncrossed and populated near the top
        best_ask = min(self.asks)
        for px in [p for p in self.bids if p >= best_ask]:
            self.bids.pop(px)
            b.append([f"{px:.1f}", "0.000"])
        if len(self.bids) < 15:
            px = round(min(self.bids) - 0.1, 1)
            self.bids[px] = self._q()
            b.append([f"{px:.1f}", f"{self.bids[px]:.3f}"])
        if len(self.asks) < 15:
            px = round(max(self.asks) + 0.1, 1)
            self.asks[px] = self._q()
            a.append([f"{px:.1f}", f"{self.asks[px]:.3f}"])
        return b, a

    def levels(self):
        bids = [[f"{p:.1f}", f"{q:.3f}"] for p, q in sorted(self.bids.items(), reverse=True)]
        asks = [[f"{p:.1f}", f"{q:.3f}"] for p, q in sorted(self.asks.items())]
        return bids, asks

    def top(self, n=10):
        bids = sorted(self.bids.items(), reverse=True)[:n]
        asks = sorted(self.asks.items())[:n]
        return bids, asks


def replay(path_root: Path, venue: str, sym: str, mult: float = 1.0):
    """Replay deltas across all day files; yield (recv_ns, valid, top10) per message."""
    files = sorted((path_root / venue / sym).glob("*/deltas.parquet"))
    book, valid, out = L2Book(), False, []
    for f in files:
        rows = pq.read_table(f).to_pylist()
        i = 0
        while i < len(rows):
            msg = rows[i]["msg"]
            group = []
            while i < len(rows) and rows[i]["msg"] == msg:
                group.append(rows[i])
                i += 1
            for r in group:
                if r["kind"] == 1:
                    book.clear()
                    valid = True
                elif r["kind"] == 2:
                    valid = False
                else:
                    book.set(r["side"], repr(r["price"]), repr(r["qty"] / mult))
            out.append((group[0]["recv_ns"], valid, book.top(10)))
    return out, files


def truth_matches(states, truth_by_recv):
    # several messages can share a receive time (e.g. a day-file reset and the update after it):
    # compare the final state at each timestamp
    final = {}
    for recv, valid, top in states:
        final[recv] = (valid, top)
    checked = 0
    for recv, (valid, top) in final.items():
        if not valid or recv not in truth_by_recv:
            continue
        tb, ta = truth_by_recv[recv]
        bids, asks = top
        assert [p for p, _ in bids] == [p for p, _ in tb]
        assert [p for p, _ in asks] == [p for p, _ in ta]
        assert all(abs(q - t) < 1e-9 for (_, q), (_, t) in zip(bids + asks, tb + ta))
        checked += 1
    return checked


def _run(tmp_path, venues):
    start, end = DAY, DAY + timedelta(days=2)
    return run(CFG, tmp_path / "raw", tmp_path / "clean", start, end, venues)


# --------------------------------------------------------------------------- Binance

def test_binance_sync_gap_disconnect_and_crosscheck(tmp_path):
    tb = TrueBook(1)
    w = RawWriter(tmp_path / "raw")
    truth, u, t = {}, 1000, T0
    w.write("binance", t, "event", '{"feed":"ws_public","event":"open"}')
    snapshot_due = {5: "sync", 400: "check", 700: "after_gap", 1100: "after_reconnect"}
    pending = []
    for i in range(1500):
        t += STEP
        b, a = tb.step()
        U, pu, u = u + 1, u, u + random.Random(i).randint(1, 5)
        if i == 600:
            continue  # dropped message: the next one's pu will not match
        if i == 1000:
            w.write("binance", t, "event", '{"feed":"ws_public","event":"close"}')
            w.write("binance", t + 1, "event", '{"feed":"ws_public","event":"open"}')
            t += 2
        d = {"e": "depthUpdate", "E": t // 1_000_000, "T": t // 1_000_000, "s": "BTCUSDT",
             "U": U, "u": u, "pu": pu, "b": b, "a": a}
        w.write("binance", t, "ws_public", orjson.dumps({"stream": "btcusdt@depth@100ms", "data": d}).decode())
        truth[t] = tb.top()
        if i in snapshot_due:
            bids, asks = tb.levels()
            # the snapshot reply arrives three messages later, as it does live
            pending.append((i + 3, {"lastUpdateId": u, "E": 0, "T": 0, "bids": bids, "asks": asks}))
        for due, snap in [p for p in pending if p[0] == i]:
            w.write("binance", t + 1, "rest_depth:BTCUSDT", orjson.dumps(snap).decode())
            pending.remove((due, snap))
        if i % 7 == 0:
            w.write("binance", t, "ws_market", orjson.dumps({"stream": "btcusdt@aggTrade", "data": {
                "e": "aggTrade", "E": 0, "a": 5000 + i, "s": "BTCUSDT", "p": "100.0", "q": "0.5",
                "f": 1, "l": 1, "T": t // 1_000_000, "m": i % 2 == 0}}).decode())
    w.flush_now()

    report = _run(tmp_path, ["binance"])["binance"]["BTCUSDT"]
    states, files = replay(tmp_path / "clean", "binance", "BTCUSDT")
    assert len(files) == 2, "data crosses midnight, so there must be two day files"
    assert truth_matches(states, truth) > 1000
    assert report["resets"] == 3          # initial sync, after the gap, after the reconnect
    assert report["gaps"] == 1 and report["pu_breaks"] == 1
    assert report["snapshot_checks"] == 1 and report["snapshot_checks_matched"] == 1
    assert report["crossed"] == 0
    trades = pq.read_table(sorted((tmp_path / "clean/binance/BTCUSDT").glob("*/trades.parquet"))[0]).to_pylist()
    assert trades[0]["side"] in (1, -1) and trades[0]["qty"] == 0.5
    # second day's file must open with a reset so it replays on its own
    second = pq.read_table(files[1]).to_pylist()
    assert second[0]["kind"] == 1


def test_binance_crosscheck_detects_bad_book(tmp_path):
    tb = TrueBook(2)
    w = RawWriter(tmp_path / "raw")
    u, t = 10, T0
    for i in range(60):
        t += STEP
        b, a = tb.step()
        U, pu, u = u + 1, u, u + 1
        d = {"e": "depthUpdate", "E": 0, "T": t // 1_000_000, "s": "BTCUSDT", "U": U, "u": u, "pu": pu, "b": b, "a": a}
        w.write("binance", t, "ws_public", orjson.dumps({"stream": "btcusdt@depth@100ms", "data": d}).decode())
        if i in (2, 40):
            bids, asks = tb.levels()
            if i == 40:
                bids[0][1] = "999.000"  # corrupt the check snapshot
            w.write("binance", t + 1, "rest_depth:BTCUSDT",
                    orjson.dumps({"lastUpdateId": u, "bids": bids, "asks": asks}).decode())
    w.flush_now()
    rep = _run(tmp_path, ["binance"])["binance"]["BTCUSDT"]
    assert rep["snapshot_checks"] == 1 and rep.get("snapshot_checks_failed") == 1


# --------------------------------------------------------------------------- Bybit

def test_bybit_snapshot_delta_gap(tmp_path):
    tb = TrueBook(3)
    w = RawWriter(tmp_path / "raw")
    truth, u, t = {}, 500, T0

    def snap(t, u):
        bids, asks = tb.levels()
        w.write("bybit", t, "ws", orjson.dumps({"topic": "orderbook.200.BTCUSDT", "type": "snapshot",
                "ts": t // 1_000_000, "data": {"s": "BTCUSDT", "b": bids, "a": asks, "u": u, "seq": 1},
                "cts": t // 1_000_000}).decode())
        truth[t] = tb.top()

    snap(t, u)
    for i in range(800):
        t += STEP
        b, a = tb.step()
        u += 1
        if i == 300:
            continue  # dropped delta
        if i == 500:
            snap(t, u)  # resubscribe snapshot after the gap
            continue
        w.write("bybit", t, "ws", orjson.dumps({"topic": "orderbook.200.BTCUSDT", "type": "delta",
                "ts": t // 1_000_000, "data": {"s": "BTCUSDT", "b": b, "a": a, "u": u, "seq": 1},
                "cts": t // 1_000_000}).decode())
        truth[t] = tb.top()
    w.flush_now()
    rep = _run(tmp_path, ["bybit"])["bybit"]["BTCUSDT"]
    states, _ = replay(tmp_path / "clean", "bybit", "BTCUSDT")
    assert truth_matches(states, truth) > 500
    assert rep["u_breaks"] == 1 and rep["resets"] == 2
    invalid = [s for s in states if not s[1]]
    assert invalid, "the gap must be marked invalid in the output"


# --------------------------------------------------------------------------- OKX

def test_okx_contracts_checksum_and_seq(tmp_path):
    tb = TrueBook(4)
    w = RawWriter(tmp_path / "raw")
    truth, seq, t = {}, 100, T0
    ref = L2Book()

    def book_msg(action, bids, asks, seq, prev, t):
        for lvl in bids:
            ref.set(BID, lvl[0], lvl[1])
        for lvl in asks:
            ref.set(ASK, lvl[0], lvl[1])
        d = {"asks": [l + ["0", "1"] for l in asks], "bids": [l + ["0", "1"] for l in bids],
             "ts": str(t // 1_000_000), "checksum": okx_checksum(ref), "seqId": seq, "prevSeqId": prev}
        w.write("okx", t, "ws", orjson.dumps({"arg": {"channel": "books", "instId": "BTC-USDT-SWAP"},
                                               "action": action, "data": [d]}).decode())

    bids, asks = tb.levels()
    book_msg("snapshot", bids, asks, seq, -1, t)
    truth[t] = tb.top()
    for i in range(500):
        t += STEP
        b, a = tb.step()
        prev, seq = seq, seq + 3
        if i == 200:
            continue  # dropped update
        if i == 250:
            ref.clear()
            bids, asks = tb.levels()
            book_msg("snapshot", bids, asks, seq, -1, t)
            truth[t] = tb.top()
            continue
        book_msg("update", b, a, seq, prev, t)
        truth[t] = tb.top()
    w.write("okx", t, "ws", orjson.dumps({"arg": {"channel": "trades", "instId": "BTC-USDT-SWAP"}, "data": [
        {"instId": "BTC-USDT-SWAP", "tradeId": "1", "px": "100.0", "sz": "250", "side": "sell", "ts": str(t // 10**6)}]}).decode())
    w.flush_now()
    rep = _run(tmp_path, ["okx"])["okx"]["BTC-USDT-SWAP"]
    # sizes are written in coins: contracts x 0.01
    states, _ = replay(tmp_path / "clean", "okx", "BTC-USDT-SWAP", mult=0.01)
    assert truth_matches(states, truth) > 200
    assert rep["seq_breaks"] == 1
    assert rep["checksums_matched"] >= 250
    trades = pq.read_table(next((tmp_path / "clean/okx/BTC-USDT-SWAP").glob("*/trades.parquet"))).to_pylist()
    assert trades[0]["qty"] == pytest.approx(2.5) and trades[0]["side"] == -1


def test_okx_checksum_format():
    b = L2Book()
    b.set(BID, "3366.1", "7")
    b.set(BID, "3366", "6")
    b.set(ASK, "3366.8", "9")
    import zlib
    expected = zlib.crc32(b"3366.1:7:3366.8:9:3366:6")
    expected = expected - (1 << 32) if expected >= (1 << 31) else expected
    assert okx_checksum(b) == expected
