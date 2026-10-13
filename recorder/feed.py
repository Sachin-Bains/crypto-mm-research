"""Shared WebSocket connection loop: connect, subscribe, read, detect stale feeds, reconnect."""

from __future__ import annotations

import asyncio
import logging
import time

import aiohttp
import orjson

from .health import Health
from .storage import RawWriter

log = logging.getLogger(__name__)


def now_ns() -> int:
    return time.time_ns()


class WsFeed:
    """One WebSocket connection.

    Raw text is written to storage before any parsing, so a parsing bug can
    never cost data. Subclasses add subscriptions, keepalive pings and
    continuity checks.
    """

    ping_interval: float | None = None
    ping_payload: str | None = None

    def __init__(
        self,
        venue: str,
        tag: str,
        url: str,
        session: aiohttp.ClientSession,
        writer: RawWriter,
        health: Health,
        idle_timeout: float = 30.0,
    ) -> None:
        self.venue = venue
        self.tag = tag
        self.url = url
        self.session = session
        self.writer = writer
        self.health = health
        self.idle_timeout = idle_timeout
        self.stats = health.feed(f"{venue}.{tag}")
        self.ws: aiohttp.ClientWebSocketResponse | None = None

    # Hooks for subclasses -------------------------------------------------
    async def on_open(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        pass

    def on_message(self, recv_ns: int, text: str) -> None:
        pass

    async def background(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        """Optional per-connection task, e.g. periodic resubscription."""
        await asyncio.Event().wait()

    def is_control(self, text: str) -> bool:
        """Keepalive replies that are not worth storing."""
        return False

    # Events recorded alongside the data ----------------------------------
    def event(self, event_name: str, /, **fields) -> None:
        payload = orjson.dumps({"feed": self.tag, "event": event_name, **fields}).decode()
        self.writer.write(self.venue, now_ns(), "event", payload)

    def gap(self, **fields) -> None:
        self.stats.gaps += 1
        self.event("gap", **fields)
        log.warning("%s.%s gap %s", self.venue, self.tag, fields)

    async def send_json(self, obj) -> None:
        if self.ws is not None and not self.ws.closed:
            await self.ws.send_str(orjson.dumps(obj).decode())

    # Main loop -----------------------------------------------------------
    async def run(self) -> None:
        backoff = 1.0
        while True:
            try:
                async with self.session.ws_connect(self.url, autoping=True, max_msg_size=0) as ws:
                    self.ws = ws
                    self.stats.connects += 1
                    self.event("open", url=self.url)
                    log.info("%s.%s connected", self.venue, self.tag)
                    backoff = 1.0
                    await self.on_open(ws)
                    helpers = [asyncio.create_task(self.background(ws))]
                    if self.ping_interval and self.ping_payload:
                        helpers.append(asyncio.create_task(self._pinger(ws)))
                    try:
                        await self._read(ws)
                    finally:
                        for t in helpers:
                            t.cancel()
                        await asyncio.gather(*helpers, return_exceptions=True)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors, handshake failures, bad status
                self.stats.errors += 1
                self.stats.last_error = repr(exc)
                log.warning("%s.%s error: %r", self.venue, self.tag, exc)
            self.ws = None
            self.stats.disconnects += 1
            self.event("close")
            log.info("%s.%s reconnecting in %.0fs", self.venue, self.tag, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def _pinger(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while not ws.closed:
            await asyncio.sleep(self.ping_interval)
            await ws.send_str(self.ping_payload)

    async def _read(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        while True:
            try:
                msg = await ws.receive(timeout=self.idle_timeout)
            except (asyncio.TimeoutError, TimeoutError):
                log.warning("%s.%s no data for %.0fs, reconnecting", self.venue, self.tag, self.idle_timeout)
                self.event("stale", seconds=self.idle_timeout)
                return
            if msg.type == aiohttp.WSMsgType.TEXT:
                recv = now_ns()
                text = msg.data
                if self.is_control(text):
                    continue
                self.writer.write(self.venue, recv, self.tag, text)
                self.stats.msgs += 1
                self.stats.bytes += len(text)
                self.stats.last_msg_ns = recv
                try:
                    self.on_message(recv, text)
                except Exception as exc:
                    self.stats.errors += 1
                    self.stats.last_error = f"parse: {exc!r}"
                    log.exception("%s.%s failed to process message", self.venue, self.tag)
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING,
                              aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                log.info("%s.%s closed by peer (%s)", self.venue, self.tag, msg.type.name)
                return


async def get_text(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> tuple[int, int, str]:
    """GET a REST endpoint. Returns (send_ns, recv_ns, body); raises on non-200."""
    t0 = now_ns()
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
        body = await resp.text()
        t1 = now_ns()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status} from {url}: {body[:200]}")
        return t0, t1, body
