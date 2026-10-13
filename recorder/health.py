"""Running counters per feed, written to a status file and summarised in the log."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FeedStats:
    msgs: int = 0
    bytes: int = 0
    last_msg_ns: int = 0
    connects: int = 0
    disconnects: int = 0
    gaps: int = 0
    snapshots: int = 0
    resyncs: int = 0
    errors: int = 0
    last_error: str = ""
    extra: dict = field(default_factory=dict)


class Health:
    def __init__(self, data_root: str | Path) -> None:
        self.feeds: dict[str, FeedStats] = {}
        self.started = time.time()
        self.data_root = Path(data_root)

    def feed(self, name: str) -> FeedStats:
        if name not in self.feeds:
            self.feeds[name] = FeedStats()
        return self.feeds[name]

    def snapshot(self) -> dict:
        now_ns = time.time_ns()
        out = {
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "uptime_s": round(time.time() - self.started),
            "feeds": {},
        }
        try:
            self.data_root.mkdir(parents=True, exist_ok=True)
            out["disk_free_gb"] = round(shutil.disk_usage(self.data_root).free / 1e9, 1)
        except OSError:
            pass
        for name, st in sorted(self.feeds.items()):
            d = asdict(st)
            d["last_msg_age_s"] = round((now_ns - st.last_msg_ns) / 1e9, 1) if st.last_msg_ns else None
            out["feeds"][name] = d
        return out

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2))
        os.replace(tmp, path)

    async def run(self, path: str | Path, status_every: float = 15.0, log_every: float = 60.0) -> None:
        prev = {n: s.msgs for n, s in self.feeds.items()}
        next_log = time.monotonic() + log_every
        try:
            while True:
                await asyncio.sleep(status_every)
                self.write(path)
                if time.monotonic() >= next_log:
                    next_log += log_every
                    for name, st in sorted(self.feeds.items()):
                        rate = (st.msgs - prev.get(name, 0)) / log_every
                        age = (time.time_ns() - st.last_msg_ns) / 1e9 if st.last_msg_ns else float("nan")
                        log.info(
                            "%-20s %7.1f msg/s  last %5.1fs ago  gaps %d  reconnects %d  errors %d",
                            name, rate, age, st.gaps, max(st.connects - 1, 0), st.errors,
                        )
                    prev = {n: s.msgs for n, s in self.feeds.items()}
        finally:
            self.write(path)
