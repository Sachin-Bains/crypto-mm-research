"""Run the recorder briefly, then verify what it captured.

Usage:  python tools/check_feeds.py --seconds 90

Checks every feed is reachable and populated, and measures the assumptions
the offline book rebuild depends on:
  * Binance: share of depth events whose pu equals the previous u
  * Bybit:   share of book deltas whose u advances by exactly one
  * OKX:     share of book updates whose prevSeqId equals the previous seqId
  * Clock offset to each venue, and feed latency after correcting for it
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.__main__ import load_config, run, setup_logging  # noqa: E402
from recorder.storage import iter_dir  # noqa: E402


def classify(venue: str, tag: str, msg) -> str:
    if tag in ("event", "clock", "meta") or tag.startswith("rest_depth"):
        return tag.split(":")[0]
    if venue == "binance":
        stream = msg.get("stream", "")
        return stream.split("@", 1)[1] if "@" in stream else "other"
    if venue == "bybit":
        topic = msg.get("topic")
        if topic:
            return f"{topic.split('.')[0]}:{msg.get('type', '')}"
        return f"op:{msg.get('op', '?')}"
    if venue == "okx":
        if "event" in msg:
            return f"event:{msg['event']}"
        ch = msg.get("arg", {}).get("channel", "?")
        return f"{ch}:{msg['action']}" if "action" in msg else ch
    return "other"


def event_time_ms(venue: str, msg) -> int | None:
    try:
        if venue == "binance":
            return int(msg["data"]["E"])
        if venue == "bybit":
            return int(msg["ts"]) if "ts" in msg else None
        if venue == "okx":
            return int(msg["data"][0]["ts"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    return None


def server_ms(venue: str, body) -> int | None:
    if venue == "binance":
        return int(body["serverTime"])
    if venue == "bybit":
        return int(body["result"]["timeNano"]) // 1_000_000
    if venue == "okx":
        return int(body["data"][0]["ts"])
    return None


def analyse(root: Path) -> bool:
    counts: dict[str, Counter] = defaultdict(Counter)
    offsets: dict[str, list[float]] = defaultdict(list)
    lat_raw: dict[str, list[float]] = defaultdict(list)

    bn_last_u, bn_ok, bn_n = {}, 0, 0
    by_last_u, by_ok, by_n = {}, 0, 0
    ok_last, ok_ok, ok_n = {}, 0, 0

    for venue, recv, tag, payload in iter_dir(root / "raw"):
        if tag in ("ws_public", "ws_market", "ws"):
            try:
                msg = orjson.loads(payload)
            except orjson.JSONDecodeError:
                counts[venue]["unparseable"] += 1
                continue
        else:
            msg = orjson.loads(payload) if tag != "event" else {}
        kind = classify(venue, tag, msg)
        counts[venue][kind] += 1

        if tag == "clock":
            srv = server_ms(venue, msg["body"])
            mid_ms = (msg["t0"] + msg["t1"]) / 2 / 1e6
            offsets[venue].append(srv - mid_ms)
            continue

        et = event_time_ms(venue, msg) if tag.startswith("ws") else None
        if et:
            lat_raw[venue].append(recv / 1e6 - et)

        if venue == "binance" and kind.startswith("depth"):
            d = msg["data"]
            prev = bn_last_u.get(d["s"])
            if prev is not None:
                bn_n += 1
                bn_ok += d["pu"] == prev
            bn_last_u[d["s"]] = d["u"]
        elif venue == "bybit" and kind.startswith("orderbook"):
            sym, u = msg["topic"].rsplit(".", 1)[1], msg["data"]["u"]
            if msg["type"] == "delta" and sym in by_last_u:
                by_n += 1
                by_ok += u == by_last_u[sym] + 1
            by_last_u[sym] = u
        elif venue == "okx" and kind.startswith("books"):
            inst = msg["arg"]["instId"]
            for d in msg["data"]:
                if msg["action"] == "update" and inst in ok_last:
                    ok_n += 1
                    ok_ok += d["prevSeqId"] == ok_last[inst]
                ok_last[inst] = d["seqId"]

    print("\n=== Messages captured ===")
    for venue in sorted(counts):
        print(f"\n{venue}")
        for kind, n in sorted(counts[venue].items()):
            print(f"  {kind:<32} {n:>8}")

    print("\n=== Clock offset (server minus local, ms) and feed latency ===")
    for venue in sorted(offsets):
        off = statistics.median(offsets[venue])
        line = f"  {venue:<8} offset {off:+8.1f}"
        if lat_raw[venue]:
            lat = [x + off for x in lat_raw[venue]]
            q = statistics.quantiles(lat, n=20)
            line += f"   latency p50 {statistics.median(lat):6.1f}  p95 {q[18]:6.1f}"
        print(line)

    print("\n=== Sequencing assumptions ===")
    def share(ok, n):
        return f"{ok}/{n} = {ok / n:.4f}" if n else "no data"
    print(f"  binance depth pu == previous u        {share(bn_ok, bn_n)}")
    print(f"  bybit book u == previous u + 1        {share(by_ok, by_n)}")
    print(f"  okx book prevSeqId == previous seqId  {share(ok_ok, ok_n)}")

    expect = {
        "binance": ["depth", "bookTicker", "aggTrade", "markPrice", "rest_depth", "clock", "meta"],
        "bybit": ["orderbook:snapshot", "orderbook:delta", "publicTrade", "tickers", "clock", "meta"],
        "okx": ["books:snapshot", "books:update", "trades", "funding-rate", "mark-price", "clock", "meta"],
    }
    print("\n=== Checks ===")
    all_ok = True
    for venue, needed in expect.items():
        if venue not in counts:
            print(f"  FAIL {venue}: nothing recorded (disabled, blocked or unreachable?)")
            all_ok = False
            continue
        for k in needed:
            got = sum(n for kind, n in counts[venue].items() if kind.startswith(k))
            status = "ok  " if got else "FAIL"
            all_ok &= bool(got)
            print(f"  {status} {venue:<8} {k:<22} {got}")
    liq = {v: sum(n for k, n in counts[v].items() if "iquidation" in k or "forceOrder" in k) for v in counts}
    print(f"\n  Liquidation messages (can be zero in a short run): {liq}")
    return all_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--root", default="data/check")
    args = ap.parse_args()

    root = Path(args.root)
    shutil.rmtree(root, ignore_errors=True)
    setup_logging(root / "logs")
    cfg = load_config(args.config)
    print(f"Recording for {args.seconds:.0f}s into {root} ...")
    asyncio.run(run(cfg, args.seconds, str(root)))
    ok = analyse(root)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED, see above and data/check/logs/recorder.log")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
