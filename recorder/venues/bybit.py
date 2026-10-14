"""Bybit v5 linear perpetuals.

One connection carries order book (snapshot then deltas), trades, tickers
(mark, index, funding) and liquidations. Bybit only sends a fresh book
snapshot on subscribe, so the recorder resubscribes the book topic on a timer
to create regular resync points without dropping the trade stream.

Bybit does not document the update id u as strictly incrementing, but live
data showed it advancing by exactly one on every delta (1803 of 1803). A jump
is therefore treated as a missed message: it is logged as a gap and the book
is resubscribed for a fresh snapshot, at most once per symbol every 60 seconds.
"""

from __future__ import annotations

import asyncio
import logging
import time

import orjson

from ..feed import WsFeed, get_text, now_ns
from ..sequencing import BybitBookTracker

log = logging.getLogger(__name__)
VENUE = "bybit"
MIN_RESUB_GAP_S = 60.0


class BybitFeed(WsFeed):
    ping_interval = 20.0
    ping_payload = '{"op":"ping"}'

    def __init__(self, cfg: dict, symbols: list[str], **kw) -> None:
        super().__init__(VENUE, "ws", cfg["url"], **kw)
        self.symbols = symbols
        self.depth = int(cfg.get("depth", 200))
        self.resync_every = float(cfg.get("resync_every_minutes", 60)) * 60
        self.book = BybitBookTracker()
        self._last_resub: dict[str, float] = {}
        self._pending: set[str] = set()

    def book_topic(self, sym: str) -> str:
        return f"orderbook.{self.depth}.{sym}"

    async def on_open(self, ws) -> None:
        for s in self.symbols:
            await self.send_json({
                "op": "subscribe",
                "args": [self.book_topic(s), f"publicTrade.{s}", f"tickers.{s}", f"allLiquidation.{s}"],
            })

    def is_control(self, text: str) -> bool:
        return '"op":"pong"' in text or '"op":"ping"' in text

    def on_message(self, recv_ns: int, text: str) -> None:
        if '"topic":"orderbook.' not in text[:80]:
            return
        msg = orjson.loads(text)
        sym = msg["topic"].rsplit(".", 1)[1]
        u = msg["data"]["u"]
        if msg.get("type") == "snapshot":
            self.book.snapshot(sym, u)
            self.stats.snapshots += 1
        elif self.book.delta(sym, u):
            self.gap(symbol=sym, kind="u_jump", u=u)
            if time.monotonic() - self._last_resub.get(sym, -1e9) >= MIN_RESUB_GAP_S:
                self._pending.add(sym)
        if self.book.deltas:
            self.stats.extra["u_step_one_share"] = round(self.book.step_one / self.book.deltas, 4)

    async def _resubscribe(self, symbols: list[str]) -> None:
        now = time.monotonic()
        for s in symbols:
            self._last_resub[s] = now
        topics = [self.book_topic(s) for s in symbols]
        await self.send_json({"op": "unsubscribe", "args": topics})
        await asyncio.sleep(0.5)
        await self.send_json({"op": "subscribe", "args": topics})
        self.stats.resyncs += 1
        self.event("resync", topics=topics)

    async def background(self, ws) -> None:
        next_timer = time.monotonic() + self.resync_every
        while True:
            await asyncio.sleep(1.0)
            if self._pending:
                syms, self._pending = sorted(self._pending), set()
                await self._resubscribe(syms)
            if time.monotonic() >= next_timer:
                next_timer += self.resync_every
                await self._resubscribe(self.symbols)


async def clock_and_meta(cfg: dict, symbols: list[str], session, writer, health, clock_every: float) -> None:
    rest = cfg["rest_url"].rstrip("/")
    stats = health.feed(f"{VENUE}.rest_meta")
    last_meta = 0.0
    while True:
        try:
            t0, t1, body = await get_text(session, f"{rest}/v5/market/time")
            writer.write(VENUE, t1, "clock", orjson.dumps({"t0": t0, "t1": t1, "body": orjson.loads(body)}).decode())
            loop_t = asyncio.get_running_loop().time()
            if last_meta == 0.0 or loop_t - last_meta > 6 * 3600:
                for s in symbols:
                    _, t1, body = await get_text(
                        session, f"{rest}/v5/market/instruments-info", {"category": "linear", "symbol": s}
                    )
                    writer.write(VENUE, t1, "meta", body)
                last_meta = loop_t
            stats.msgs += 1
            stats.last_msg_ns = now_ns()
        except Exception as exc:
            stats.errors += 1
            stats.last_error = repr(exc)
            log.warning("bybit clock/meta failed: %r", exc)
        await asyncio.sleep(clock_every)


def build(cfg: dict, symbols: list[str], session, writer, health, clock_every: float):
    return [
        BybitFeed(cfg, symbols, session=session, writer=writer, health=health).run(),
        clock_and_meta(cfg, symbols, session, writer, health, clock_every),
    ]
