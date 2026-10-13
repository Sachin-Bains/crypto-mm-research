"""Entry point: python -m recorder --config config.toml"""

from __future__ import annotations

import argparse
import asyncio
import logging
import logging.handlers
import sys
import tomllib
from pathlib import Path

import aiohttp

from .health import Health
from .storage import RawWriter
from .venues import binance, bybit, okx

log = logging.getLogger("recorder")
BUILDERS = {"binance": binance.build, "bybit": bybit.build, "okx": okx.build}


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def setup_logging(log_dir: Path, verbose: bool = False) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file = logging.handlers.RotatingFileHandler(
        log_dir / "recorder.log", maxBytes=20_000_000, backupCount=5, encoding="utf-8"
    )
    file.setFormatter(fmt)
    root.addHandler(console)
    root.addHandler(file)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


async def run(cfg: dict, duration: float | None = None, root_override: str | None = None) -> Health:
    root = Path(root_override or cfg["storage"]["root"])
    writer = RawWriter(root / "raw", level=int(cfg["storage"].get("zstd_level", 3)))
    health = Health(root)
    clock_every = float(cfg.get("clock", {}).get("every_seconds", 60))

    timeout = aiohttp.ClientTimeout(total=None, connect=15, sock_connect=15)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        coros = []
        for venue, build in BUILDERS.items():
            vcfg = cfg.get(venue, {})
            if not vcfg.get("enabled", False):
                continue
            symbols = cfg["symbols"][venue]
            coros += build(vcfg, symbols, session, writer, health, clock_every)
            log.info("%s enabled for %s", venue, ", ".join(symbols))
        if not coros:
            raise SystemExit("No venues enabled in config")

        tasks = [asyncio.create_task(c) for c in coros]
        tasks.append(asyncio.create_task(writer.run(float(cfg["storage"].get("flush_seconds", 2.0)))))
        tasks.append(asyncio.create_task(health.run(root / "status.json")))
        try:
            if duration:
                await asyncio.sleep(duration)
            else:
                await asyncio.gather(*tasks)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.flush_now()
            health.write(root / "status.json")
            log.info("Stopped. %.1f MB compressed written this session.", writer.bytes_written / 1e6)
    return health


def main() -> None:
    ap = argparse.ArgumentParser(description="Record raw crypto perp market data")
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--duration", type=float, default=None, help="Stop after N seconds (default: run forever)")
    ap.add_argument("--root", default=None, help="Override storage root")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.root or cfg["storage"]["root"])
    setup_logging(root / "logs", args.verbose)
    try:
        asyncio.run(run(cfg, args.duration, args.root))
    except KeyboardInterrupt:
        log.info("Interrupted, data flushed.")


if __name__ == "__main__":
    main()
