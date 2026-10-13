"""Append-only raw message storage.

Every message is stored exactly as the exchange sent it, next to the local
receive time. Book reconstruction happens offline, so a bug in the rebuild
logic can always be fixed and re-run against untouched raw data.

File layout: <root>/<venue>/<YYYY-MM-DD>/<HH>.ndjson.zst  (UTC hours)
Line format: <recv_ns>\t<tag>\t<payload>\n

Each flush appends one independent zstd frame. A crash or power cut loses
at most the last flush interval and never corrupts data already on disk.
"""

from __future__ import annotations

import asyncio
import io
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import zstandard as zstd

log = logging.getLogger(__name__)

NS_PER_HOUR = 3_600_000_000_000


def _hour_parts(hour_index: int) -> tuple[str, str]:
    dt = datetime.fromtimestamp(hour_index * 3600, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.strftime("%H")


class RawWriter:
    def __init__(self, root: str | Path, level: int = 3) -> None:
        self.root = Path(root)
        self.level = level
        self._buf: dict[tuple[str, int], list[str]] = defaultdict(list)
        self.bytes_written = 0

    def write(self, venue: str, recv_ns: int, tag: str, payload: str) -> None:
        # Exchange JSON is compact; any raw newline can only be insignificant
        # whitespace, so stripping it keeps one message per line safely.
        if "\n" in payload or "\r" in payload:
            payload = payload.replace("\r", "").replace("\n", "")
        self._buf[(venue, recv_ns // NS_PER_HOUR)].append(f"{recv_ns}\t{tag}\t{payload}\n")

    def take(self) -> dict[tuple[str, int], list[str]]:
        """Swap out the current buffer. Called on the event loop thread."""
        batch, self._buf = self._buf, defaultdict(list)
        return batch

    def write_batch(self, batch: dict[tuple[str, int], list[str]]) -> None:
        """Compress and append a batch. Safe to run in a worker thread."""
        cctx = zstd.ZstdCompressor(level=self.level)
        for (venue, hour_index), lines in batch.items():
            if not lines:
                continue
            day, hour = _hour_parts(hour_index)
            path = self.root / venue / day / f"{hour}.ndjson.zst"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = cctx.compress("".join(lines).encode("utf-8"))
            with open(path, "ab") as fh:
                fh.write(frame)
            self.bytes_written += len(frame)

    def flush_now(self) -> None:
        self.write_batch(self.take())

    async def run(self, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                await asyncio.to_thread(self.write_batch, self.take())
        finally:
            self.flush_now()


def iter_file(path: str | Path) -> Iterator[tuple[int, str, str]]:
    """Yield (recv_ns, tag, payload) from one raw file.

    Tolerates a truncated final frame (from a crash) by stopping cleanly at
    the last complete line.
    """
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as fh:
        reader = dctx.stream_reader(fh, read_across_frames=True)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="\n")
        try:
            for line in text:
                if not line.endswith("\n"):
                    log.warning("Truncated line at end of %s, stopping", path)
                    return
                recv, tag, payload = line[:-1].split("\t", 2)
                yield int(recv), tag, payload
        except zstd.ZstdError as exc:
            log.warning("Incomplete zstd frame at end of %s (%s), stopping", path, exc)


def iter_dir(root: str | Path, venue: str | None = None) -> Iterator[tuple[str, int, str, str]]:
    """Yield (venue, recv_ns, tag, payload) across all files, in file order."""
    root = Path(root)
    venues = [venue] if venue else sorted(p.name for p in root.iterdir() if p.is_dir())
    for v in venues:
        for path in sorted((root / v).glob("*/*.ndjson.zst")):
            for recv, tag, payload in iter_file(path):
                yield v, recv, tag, payload
