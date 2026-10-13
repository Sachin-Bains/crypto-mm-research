"""Offline tests: storage, sequencing rules, and the connection loop against a fake exchange."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import aiohttp
import orjson
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.health import Health  # noqa: E402
from recorder.sequencing import BinanceDepthTracker, BybitBookTracker, IdTracker, OkxBookTracker  # noqa: E402
from recorder.storage import NS_PER_HOUR, RawWriter, iter_dir, iter_file  # noqa: E402
from recorder.venues import binance  # noqa: E402
from recorder.feed import WsFeed  # noqa: E402

T0 = 1_760_000_000 * 1_000_000_000  # a fixed UTC instant


# Storage ------------------------------------------------------------------

def test_storage_round_trip_across_hours(tmp_path):
    w = RawWriter(tmp_path)
    w.write("binance", T0, "ws_public", '{"a":1}')
    w.write("binance", T0 + NS_PER_HOUR, "ws_public", '{"a":2}')
    w.flush_now()
    w.write("binance", T0 + 1, "ws_market", '{"a":3,\n"b":4}')  # second frame, same file
    w.flush_now()
    files = sorted((tmp_path / "binance").glob("*/*.ndjson.zst"))
    assert len(files) == 2
    rows = list(iter_dir(tmp_path))
    assert [r[3] for r in rows] == ['{"a":1}', '{"a":3,"b":4}', '{"a":2}']
    assert rows[0][1] == T0 and rows[1][2] == "ws_market"


def test_storage_survives_truncated_frame(tmp_path):
    w = RawWriter(tmp_path)
    for i in range(100):
        w.write("okx", T0 + i, "ws", f'{{"i":{i}}}')
    w.flush_now()
    for i in range(100, 200):
        w.write("okx", T0 + i, "ws", f'{{"i":{i}}}')
    w.flush_now()
    path = next((tmp_path / "okx").glob("*/*.ndjson.zst"))
    data = path.read_bytes()
    path.write_bytes(data[:-15])  # simulate power cut mid-write
    rows = list(iter_file(path))
    assert len(rows) >= 100  # first frame fully intact
    assert all(r[2] == f'{{"i":{k}}}' for k, r in enumerate(rows))


# Sequencing ---------------------------------------------------------------

def test_binance_pu_rule():
    t = BinanceDepthTracker()
    assert not t.check("BTCUSDT", u=10, pu=5)    # first event, nothing to compare
    assert not t.check("BTCUSDT", u=15, pu=10)
    assert t.check("BTCUSDT", u=30, pu=20)        # missed the event ending at 20
    assert not t.check("ETHUSDT", u=3, pu=1)      # symbols tracked separately


def test_id_tracker_counts_missing():
    t = IdTracker()
    assert t.check("BTCUSDT", 100) == 0
    assert t.check("BTCUSDT", 101) == 0
    assert t.check("BTCUSDT", 105) == 3


def test_bybit_tracker_measures_steps():
    t = BybitBookTracker()
    t.snapshot("BTCUSDT", 1000)
    assert not t.delta("BTCUSDT", 1001)
    assert t.delta("BTCUSDT", 1005)
    assert t.step_one == 1 and t.deltas == 2


def test_okx_seq_rule():
    t = OkxBookTracker()
    t.snapshot("BTC-USDT-SWAP", 50)
    assert not t.update("BTC-USDT-SWAP", prev_seq=50, seq=57)
    assert not t.update("BTC-USDT-SWAP", prev_seq=57, seq=57)  # no-change heartbeat
    assert t.update("BTC-USDT-SWAP", prev_seq=60, seq=61)


# Connection loop against a fake exchange ------------------------------------

def depth(sym, U, u, pu):
    return orjson.dumps({"stream": f"{sym.lower()}@depth@100ms",
                         "data": {"e": "depthUpdate", "E": 1, "s": sym, "U": U, "u": u, "pu": pu,
                                  "b": [["100.0", "1.0"]], "a": []}}).decode()


async def _fake_binance(snapshot_calls: list, connections: list):
    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        connections.append(request.query.get("streams"))
        if len(connections) == 1:
            await ws.send_str(depth("BTCUSDT", 1, 10, 0))
            await ws.send_str(depth("BTCUSDT", 11, 20, 10))
            await ws.send_str(depth("BTCUSDT", 31, 40, 30))  # gap: pu 30 != previous u 20
            await asyncio.sleep(0.2)
            await ws.close()  # force a reconnect
        else:
            await ws.send_str(depth("BTCUSDT", 41, 50, 40))
            await asyncio.sleep(5)
        return ws

    async def depth_handler(request):
        snapshot_calls.append(request.query["symbol"])
        return web.json_response({"lastUpdateId": 15, "E": 1, "T": 1, "bids": [], "asks": []})

    app = web.Application()
    app.router.add_get("/public/stream", ws_handler)
    app.router.add_get("/fapi/v1/depth", depth_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def test_binance_feed_gap_snapshot_and_reconnect(tmp_path):
    async def scenario():
        snapshot_calls, connections = [], []
        runner, port = await _fake_binance(snapshot_calls, connections)
        cfg = {"public_url": f"ws://127.0.0.1:{port}/public/stream",
               "rest_url": f"http://127.0.0.1:{port}", "snapshot_every_minutes": 60}
        writer, health = RawWriter(tmp_path), Health(tmp_path)
        async with aiohttp.ClientSession() as session:
            snap = binance.DepthSnapshotter(cfg, ["BTCUSDT"], session, writer, health)
            feed = binance.BinancePublic(cfg, ["BTCUSDT"], snap, session=session, writer=writer, health=health)
            tasks = [asyncio.create_task(feed.run()), asyncio.create_task(snap.run())]
            await asyncio.sleep(4.5)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        writer.flush_now()
        return feed, snapshot_calls, connections

    feed, snapshot_calls, connections = asyncio.run(scenario())
    assert len(connections) == 2, "feed should reconnect after the server closes"
    assert feed.stats.gaps == 1
    assert feed.stats.msgs == 4
    # first connect + gap coalesce into one fetch; reconnect triggers another
    assert len(snapshot_calls) >= 2
    tags = [r[2] for r in iter_dir(tmp_path)]
    assert tags.count("ws_public") == 4
    assert tags.count("rest_depth:BTCUSDT") == len(snapshot_calls)
    events = [orjson.loads(r[3])["event"] for r in iter_dir(tmp_path) if r[2] == "event"]
    assert events.count("open") == 2 and "gap" in events and "close" in events


def test_stale_feed_reconnects(tmp_path):
    """A connection that stays open but sends nothing must be dropped and retried."""
    async def scenario():
        hits = []

        async def ws_handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            hits.append(1)
            await asyncio.sleep(10)
            return ws

        app = web.Application()
        app.router.add_get("/ws", ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        writer, health = RawWriter(tmp_path), Health(tmp_path)
        async with aiohttp.ClientSession() as session:
            feed = WsFeed("test", "ws", f"ws://127.0.0.1:{port}/ws", session, writer, health, idle_timeout=0.5)
            task = asyncio.create_task(feed.run())
            await asyncio.sleep(3.0)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()
        return hits

    assert len(asyncio.run(scenario())) >= 2
