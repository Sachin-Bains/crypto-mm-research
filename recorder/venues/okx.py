"""OKX v5 USDT-margined swaps.

The `books` channel sends a 400-level snapshot on subscribe, then updates
carrying seqId and prevSeqId: an update is continuous if its prevSeqId equals
the seqId of the message before it. On a detected gap the book is
resubscribed for a fresh snapshot, rate-limited so a misbehaving feed can
never burn through OKX's 480 subscribe requests per hour. The book is also
resubscribed on a timer for regular resync points.

Sizes on OKX swaps are in contracts, not coins. The contract value (ctVal)
is stored with the instrument metadata so the offline rebuild can convert.
"""

from __future__ import annotations

import asyncio
import logging
import time

import orjson

from ..feed import WsFeed, get_text, now_ns
from ..sequencing import OkxBookTracker

log = logging.getLogger(__name__)
VENUE = "okx"
MIN_RESUB_GAP_S = 300.0


class OkxFeed(WsFeed):
    ping_interval = 20.0
    ping_payload = "ping"

    def __init__(self, cfg: dict, symbols: list[str], **kw) -> None:
        super().__init__(VENUE, "ws", cfg["url"], **kw)
        self.symbols = symbols
        self.book_channel = cfg.get("book_channel", "books")
        self.resync_every = float(cfg.get("resync_every_minutes", 60)) * 60
        self.book = OkxBookTracker()
        self._last_resub: dict[str, float] = {}
        self._pending: set[str] = set()

    def book_args(self, insts: list[str]) -> list[dict]:
        return [{"channel": self.book_channel, "instId": i} for i in insts]

    async def on_open(self, ws) -> None:
        args = []
        for i in self.symbols:
            args += [
                {"channel": self.book_channel, "instId": i},
                {"channel": "trades", "instId": i},
                {"channel": "funding-rate", "instId": i},
                {"channel": "mark-price", "instId": i},
            ]
        args.append({"channel": "liquidation-orders", "instType": "SWAP"})
        await self.send_json({"op": "subscribe", "args": args})

    def is_control(self, text: str) -> bool:
        return text == "pong"

    def on_message(self, recv_ns: int, text: str) -> None:
        if f'"channel":"{self.book_channel}"' not in text[:120] or '"data"' not in text:
            return
        msg = orjson.loads(text)
        inst = msg["arg"]["instId"]
        action = msg.get("action")
        for d in msg["data"]:
            seq, prev = d.get("seqId"), d.get("prevSeqId")
            if seq is None:
                continue
            if action == "snapshot":
                self.book.snapshot(inst, seq)
                self.stats.snapshots += 1
            elif self.book.update(inst, prev, seq):
                self.gap(symbol=inst, kind="seqId", seqId=seq, prevSeqId=prev)
                self._request_resub(inst)

    def _request_resub(self, inst: str) -> None:
        if time.monotonic() - self._last_resub.get(inst, -1e9) >= MIN_RESUB_GAP_S:
            self._pending.add(inst)

    async def _resubscribe(self, insts: list[str]) -> None:
        now = time.monotonic()
        for i in insts:
            self._last_resub[i] = now
        await self.send_json({"op": "unsubscribe", "args": self.book_args(insts)})
        await asyncio.sleep(0.5)
        await self.send_json({"op": "subscribe", "args": self.book_args(insts)})
        self.stats.resyncs += 1
        self.event("resync", insts=insts)

    async def background(self, ws) -> None:
        next_timer = time.monotonic() + self.resync_every
        while True:
            await asyncio.sleep(1.0)
            if self._pending:
                insts, self._pending = sorted(self._pending), set()
                await self._resubscribe(insts)
            if time.monotonic() >= next_timer:
                next_timer += self.resync_every
                await self._resubscribe(self.symbols)


async def clock_and_meta(cfg: dict, symbols: list[str], session, writer, health, clock_every: float) -> None:
    rest = cfg["rest_url"].rstrip("/")
    stats = health.feed(f"{VENUE}.rest_meta")
    last_meta = 0.0
    while True:
        try:
            t0, t1, body = await get_text(session, f"{rest}/api/v5/public/time")
            writer.write(VENUE, t1, "clock", orjson.dumps({"t0": t0, "t1": t1, "body": orjson.loads(body)}).decode())
            loop_t = asyncio.get_running_loop().time()
            if last_meta == 0.0 or loop_t - last_meta > 6 * 3600:
                for i in symbols:
                    _, t1, body = await get_text(
                        session, f"{rest}/api/v5/public/instruments", {"instType": "SWAP", "instId": i}
                    )
                    writer.write(VENUE, t1, "meta", body)
                last_meta = loop_t
            stats.msgs += 1
            stats.last_msg_ns = now_ns()
        except Exception as exc:
            stats.errors += 1
            stats.last_error = repr(exc)
            log.warning("okx clock/meta failed: %r", exc)
        await asyncio.sleep(clock_every)


def build(cfg: dict, symbols: list[str], session, writer, health, clock_every: float):
    return [
        OkxFeed(cfg, symbols, session=session, writer=writer, health=health).run(),
        clock_and_meta(cfg, symbols, session, writer, health, clock_every),
    ]
