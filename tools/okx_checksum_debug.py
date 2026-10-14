"""Work out which checksum rule matches OKX's live books data.

Usage:  python tools/okx_checksum_debug.py --date 2026-10-13

Replays the first few thousand OKX book messages, computes the checksum under
several candidate rules after each one, and prints how often each matches the
exchange's own value, plus a sample of the raw fields for inspection.
"""

from __future__ import annotations

import argparse
import sys
import zlib
from collections import Counter
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuild.book import ASK, BID, L2Book  # noqa: E402
from recorder.storage import iter_file  # noqa: E402


def signed(c: int) -> int:
    return c - (1 << 32) if c >= (1 << 31) else c


def build(book: L2Book, n: int, bid_first: bool, sep: str = ":") -> str:
    bids, asks = book.top_strings(n)
    parts: list[str] = []
    for i in range(n):
        first, second = (bids, asks) if bid_first else (asks, bids)
        if i < len(first):
            parts += first[i]
        if i < len(second):
            parts += second[i]
    return sep.join(parts)


def build_float(book: L2Book, n: int) -> str:
    bids, asks = book.top(n)
    parts: list[str] = []
    for i in range(n):
        if i < len(bids):
            parts += [repr(bids[i][0]), repr(bids[i][1])]
        if i < len(asks):
            parts += [repr(asks[i][0]), repr(asks[i][1])]
    return ":".join(parts)


VARIANTS = {
    "bid first, 25 levels (current)": lambda b: build(b, 25, True),
    "ask first, 25 levels": lambda b: build(b, 25, False),
    "bid first, 5 levels": lambda b: build(b, 5, True),
    "bid first, 400 levels": lambda b: build(b, 400, True),
    "bid first, parsed floats": lambda b: build_float(b, 25),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--inst", default="BTC-USDT-SWAP")
    ap.add_argument("--messages", type=int, default=3000)
    args = ap.parse_args()

    files = sorted(Path("data/raw/okx", args.date).glob("*.ndjson.zst"))
    if not files:
        sys.exit(f"No OKX raw files for {args.date}")

    book, synced, last_seq, n = L2Book(), False, None, 0
    hits = Counter()
    unsigned_hits = Counter()
    checksum_values = Counter()
    samples = []
    for path in files:
        for _, tag, payload in iter_file(path):
            if tag != "ws" or '"channel":"books"' not in payload[:80] or '"data"' not in payload:
                continue
            msg = orjson.loads(payload)
            if msg["arg"].get("instId") != args.inst:
                continue
            for d in msg["data"]:
                if msg.get("action") == "snapshot":
                    book.load(d["bids"], d["asks"])
                    synced, last_seq = True, d["seqId"]
                    if len(samples) < 1:
                        samples.append({"keys": sorted(d.keys()), "first_bid": d["bids"][:2],
                                        "first_ask": d["asks"][:2], "checksum": d.get("checksum")})
                elif synced and d["prevSeqId"] == last_seq:
                    for lvl in d["bids"]:
                        book.set(BID, lvl[0], lvl[1])
                    for lvl in d["asks"]:
                        book.set(ASK, lvl[0], lvl[1])
                    last_seq = d["seqId"]
                else:
                    synced = False
                    continue
                expected = d.get("checksum")
                checksum_values[expected] += 1
                if expected is None:
                    continue
                n += 1
                for name, fn in VARIANTS.items():
                    crc = zlib.crc32(fn(book).encode())
                    hits[name] += signed(crc) == int(expected)
                    unsigned_hits[name] += crc == int(expected)
            if n >= args.messages:
                break
        if n >= args.messages:
            break

    print(f"\n{args.inst}: {n} book messages checked")
    print(f"Distinct checksum values seen: {len(checksum_values)}  (most common: {checksum_values.most_common(3)})")
    print("\nMatch rate by rule (signed / unsigned CRC):")
    for name in VARIANTS:
        print(f"  {name:<34} {hits[name] / max(n, 1):7.2%}   {unsigned_hits[name] / max(n, 1):7.2%}")
    print("\nFirst snapshot sample:")
    for s in samples:
        print("  fields:", s["keys"])
        print("  bids:", s["first_bid"], " asks:", s["first_ask"], " checksum:", s["checksum"])
    print("\nCurrent rule string (first 160 chars):")
    print(" ", VARIANTS["bid first, 25 levels (current)"](book)[:160])


if __name__ == "__main__":
    main()
