"""Rebuild validated order books and trades from raw recordings.

Usage:
  python -m rebuild --start 2026-10-14 --end 2026-10-14
  python -m rebuild --start 2026-10-14 --end 2026-10-20 --venues binance,okx

Reads data/raw, writes data/clean/<venue>/<symbol>/<date>/{deltas,top,trades,bbo}.parquet
and a quality report to data/clean/reports/. Processing starts two hours before
--start so every book is already in sync when the requested window begins.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from recorder.__main__ import load_config
from recorder.storage import iter_file

from .venues import VENUES

log = logging.getLogger("rebuild")
WARMUP = timedelta(hours=2)


def parse_day(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def hour_files(raw_root: Path, venue: str, t0: datetime, t1: datetime) -> list[Path]:
    files = []
    for path in sorted((raw_root / venue).glob("*/*.ndjson.zst")):
        try:
            t = datetime.strptime(f"{path.parent.name} {path.name[:2]}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if t0 <= t < t1:
            files.append(path)
    return files


def run(cfg: dict, raw_root: Path, out_root: Path, start: datetime, end_excl: datetime, venues: list[str]) -> dict:
    start_ns, end_ns = int(start.timestamp() * 1e9), int(end_excl.timestamp() * 1e9)
    report = {}
    for venue in venues:
        files = hour_files(raw_root, venue, start - WARMUP, end_excl)
        if not files:
            log.warning("%s: no raw files in range", venue)
            continue
        proc = VENUES[venue](out_root, cfg["symbols"][venue], start_ns, end_ns)
        t_wall, n, last = time.time(), 0, start_ns
        for path in files:
            for recv, tag, payload in iter_file(path):
                if recv >= end_ns:
                    break
                proc.handle(recv, tag, payload)
                n += 1
                last = recv
            log.info("%s %s/%s done (%d messages so far)", venue, path.parent.name, path.name[:2], n)
        report[venue] = proc.finish(last)
        log.info("%s: %d messages in %.0fs", venue, n, time.time() - t_wall)
    return report


def print_report(report: dict) -> None:
    print("\n=== Rebuild quality ===")
    for venue, syms in report.items():
        for sym, s in syms.items():
            print(f"\n{venue} {sym}")
            print(f"  valid {s['valid_hours']:.2f} of {s['data_hours']:.2f} h recorded ({s['valid_share']:.1%})   "
                  f"resets {s['resets']}   invalid episodes {s['invalid_episodes']} (gaps {s['gaps']})")
            print(f"  book messages {s['book_msgs']}   crossed after update {s['crossed']}   "
                  f"trades {s['trades']}   bbo {s['bbo']}")
            for key in ("snapshot_checks", "checksums"):
                if s.get(key):
                    ok = s.get(f"{key}_matched", 0)
                    print(f"  {key.replace('_', ' '):<16} {ok}/{s[key]} matched ({ok / s[key]:.2%})")
            if s.get("checksum_not_provided"):
                print(f"  checksum         not provided by the exchange ({s['checksum_not_provided']} messages)")
            other = {k: v for k, v in s.items() if k.endswith(("breaks", "skipped", "stale_snapshots", "failed"))}
            if other:
                print("  " + "  ".join(f"{k} {v}" for k, v in other.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--start", required=True, help="first UTC date, YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="last UTC date, inclusive (default: same as --start)")
    ap.add_argument("--venues", default="binance,bybit,okx")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/clean")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    cfg = load_config(args.config)
    start = parse_day(args.start)
    end_excl = parse_day(args.end or args.start) + timedelta(days=1)
    venues = [v.strip() for v in args.venues.split(",") if v.strip()]
    report = run(cfg, Path(args.raw), Path(args.out), start, end_excl, venues)
    print_report(report)

    rep_dir = Path(args.out) / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    path = rep_dir / f"rebuild_{args.start}_{args.end or args.start}.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {path}")


if __name__ == "__main__":
    main()
