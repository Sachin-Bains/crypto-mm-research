"""Summarise recorded raw data: volume per venue and tag, time span, disk use.

Usage:  python tools/inspect_raw.py                 (everything under data/raw)
        python tools/inspect_raw.py --date 2026-10-13
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder.storage import iter_file  # noqa: E402


def fmt_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    pattern = f"{args.date}/*.ndjson.zst" if args.date else "*/*.ndjson.zst"
    for venue_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(venue_dir.glob(pattern))
        if not files:
            continue
        tags: Counter = Counter()
        events: Counter = Counter()
        first = last = None
        disk = 0
        for f in files:
            disk += f.stat().st_size
            for recv, tag, payload in iter_file(f):
                tags[tag] += 1
                if tag == "event":
                    for key in ("gap", "open", "close", "stale", "resync"):
                        if f'"event":"{key}"' in payload:
                            events[key] += 1
                first = recv if first is None else min(first, recv)
                last = recv if last is None else max(last, recv)
        span_h = (last - first) / 3.6e12 if first else 0
        print(f"\n{venue_dir.name}: {len(files)} files, {disk / 1e6:.1f} MB on disk, {span_h:.2f} h")
        print(f"  {fmt_ns(first)}  ->  {fmt_ns(last)} UTC")
        for tag, n in sorted(tags.items()):
            print(f"  {tag:<22} {n:>10}")
        if events:
            print("  events: " + ", ".join(f"{k} {v}" for k, v in sorted(events.items())))


if __name__ == "__main__":
    main()
