"""Binance USD-M perpetual futures.

Since the 2026 URL split, high-frequency book data lives on /public and
trades, mark price and liquidations on /market. A client still on the old
/stream URL keeps getting depth but silently loses the /market channels.

Book sync (done offline from the raw data) follows Binance's documented
procedure: REST snapshot with lastUpdateId, discard events with
u < lastUpdateId, first applied event has U <= lastUpdateId <= u, then every
event's pu must equal the previous u. The recorder therefore stores REST
snapshots alongside the stream: at every connect, on every detected gap,
and on a timer so there is always a recent resync point.
"""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import orjson

from ..feed import WsFeed, get_text, now_ns
from ..sequencing import BinanceDepthTracker, IdTracker

log = logging.getLogger(__name__)
VENUE = "binance"


class DepthSnapshotter:
    """Fetches and stores REST depth snapshots, on demand and on a timer."""

    def __init__(self, cfg: dict, symbols: list[str], session, writer, health) -> None:
        self.rest = cfg["rest_url"].rstrip("/")
        self.limit = int(cfg.get("snapshot_limit", 1000))
        self.every = float(cfg.get("snapshot_every_minutes", 30)) * 60
        self.symbols = symbols
        self.session = session
        self.writer = writer
        self.stats = health.feed(f"{VENUE}.rest_depth")
        self._wanted: dict[str, asyncio.Event] = {s: asyncio.Event() for s in symbols}

    def request(self, symbol: str) -> None:
        self._wanted[symbol].set()

    def request_all(self) -> None:
        for ev in self._wanted.values():
            ev.set()

    async def _fetch(self, symbol: str) -> None:
        try:
            _, t1, body = await get_text(
                self.session, f"{self.rest}/fapi/v1/depth", {"symbol": symbol, "limit": self.limit}
            )
            self.writer.write(VENUE, t1, f"rest_depth:{symbol}", body)
            self.stats.snapshots += 1
            self.stats.msgs += 1
            self.stats.last_msg_ns = t1
        except Exception as exc:
            self.stats.errors += 1
            self.stats.last_error = repr(exc)
            log.warning("binance depth snapshot %s failed: %r", symbol, exc)

    async def _loop(self, symbol: str) -> None:
        ev = self._wanted[symbol]
        while True:
            try:
                await asyncio.wait_for(ev.wait(), timeout=self.every)
            except (asyncio.TimeoutError, TimeoutError):
                pass
            ev.clear()
            # Short pause so the stream has buffered events that straddle the snapshot.
            await asyncio.sleep(1.0)
            await self._fetch(symbol)

    async def run(self) -> None:
        await asyncio.gather(*(self._loop(s) for s in self.symbols))


class BinancePublic(WsFeed):
    def __init__(self, cfg: dict, symbols: list[str], snapshotter: DepthSnapshotter, **kw) -> None:
        speed = cfg.get("depth_speed", "100ms")
        streams = [f"{s.lower()}@depth@{speed}" for s in symbols]
        if cfg.get("book_ticker", True):
            streams += [f"{s.lower()}@bookTicker" for s in symbols]
        url = f"{cfg['public_url']}?streams={'/'.join(streams)}"
        super().__init__(VENUE, "ws_public", url, **kw)
        self.snapshotter = snapshotter
        self.depth = BinanceDepthTracker()

    async def on_open(self, ws) -> None:
        self.depth.reset()
        self.snapshotter.request_all()

    def on_message(self, recv_ns: int, text: str) -> None:
        if "@depth" not in text[:60]:
            return
        d = orjson.loads(text)["data"]
        sym = d["s"]
        if self.depth.check(sym, d["u"], d["pu"]):
            self.gap(symbol=sym, kind="depth_pu", u=d["u"], pu=d["pu"])
            self.snapshotter.request(sym)


class BinanceMarket(WsFeed):
    def __init__(self, cfg: dict, symbols: list[str], **kw) -> None:
        streams = []
        for s in symbols:
            s = s.lower()
            streams += [f"{s}@aggTrade", f"{s}@markPrice@1s", f"{s}@forceOrder"]
        url = f"{cfg['market_url']}?streams={'/'.join(streams)}"
        super().__init__(VENUE, "ws_market", url, **kw)
        self.trades = IdTracker()

    async def on_open(self, ws) -> None:
        self.trades.reset()

    def on_message(self, recv_ns: int, text: str) -> None:
        if "@aggTrade" not in text[:60]:
            return
        d = orjson.loads(text)["data"]
        missing = self.trades.check(d["s"], d["a"])
        if missing:
            self.gap(symbol=d["s"], kind="aggTrade_id", missing=missing)


async def clock_and_meta(cfg: dict, symbols: list[str], session, writer, health, clock_every: float) -> None:
    """Server-time samples for clock offset, plus instrument metadata every 6 hours."""
    rest = cfg["rest_url"].rstrip("/")
    stats = health.feed(f"{VENUE}.rest_meta")
    last_meta = 0.0
    while True:
        try:
            t0, t1, body = await get_text(session, f"{rest}/fapi/v1/time")
            writer.write(VENUE, t1, "clock", orjson.dumps({"t0": t0, "t1": t1, "body": orjson.loads(body)}).decode())
            if asyncio.get_running_loop().time() - last_meta > 6 * 3600 or last_meta == 0.0:
                _, t1, body = await get_text(session, f"{rest}/fapi/v1/exchangeInfo")
                info = orjson.loads(body)
                keep = [x for x in info.get("symbols", []) if x.get("symbol") in symbols]
                writer.write(VENUE, t1, "meta", orjson.dumps({"symbols": keep}).decode())
                last_meta = asyncio.get_running_loop().time()
            stats.msgs += 1
            stats.last_msg_ns = now_ns()
        except Exception as exc:
            stats.errors += 1
            stats.last_error = repr(exc)
            log.warning("binance clock/meta failed: %r", exc)
        await asyncio.sleep(clock_every)


def build(cfg: dict, symbols: list[str], session: aiohttp.ClientSession, writer, health, clock_every: float):
    snap = DepthSnapshotter(cfg, symbols, session, writer, health)
    kw = dict(session=session, writer=writer, health=health)
    return [
        BinancePublic(cfg, symbols, snap, **kw).run(),
        BinanceMarket(cfg, symbols, **kw).run(),
        snap.run(),
        clock_and_meta(cfg, symbols, session, writer, health, clock_every),
    ]
