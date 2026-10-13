"""Online continuity checks for each venue's feeds.

These do not rebuild the book. They only answer "did we just miss a
message?", so the recorder can log the gap and fetch a fresh snapshot.
Each rule comes from the venue's documented sequencing fields.
"""

from __future__ import annotations


class BinanceDepthTracker:
    """Binance USD-M diff depth: each event's `pu` must equal the previous event's `u`."""

    def __init__(self) -> None:
        self.last_u: dict[str, int] = {}

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self.last_u.clear()
        else:
            self.last_u.pop(symbol, None)

    def check(self, symbol: str, u: int, pu: int) -> bool:
        """Return True if a gap was detected before this event."""
        prev = self.last_u.get(symbol)
        self.last_u[symbol] = u
        return prev is not None and pu != prev


class IdTracker:
    """Strictly incrementing ids, e.g. Binance aggTrade `a`. Returns the number of missing ids."""

    def __init__(self) -> None:
        self.last: dict[str, int] = {}

    def reset(self) -> None:
        self.last.clear()

    def check(self, key: str, value: int) -> int:
        prev = self.last.get(key)
        self.last[key] = value
        if prev is None or value <= prev:
            return 0
        return value - prev - 1


class BybitBookTracker:
    """Bybit v5 orderbook: a snapshot resets state; deltas carry update id `u`.

    Bybit does not document that `u` increments by exactly one, so jumps are
    counted for inspection rather than treated as proven gaps. The first live
    run measures this (see tools/check_feeds.py).
    """

    def __init__(self) -> None:
        self.last_u: dict[str, int] = {}
        self.deltas = 0
        self.step_one = 0

    def snapshot(self, symbol: str, u: int) -> None:
        self.last_u[symbol] = u

    def delta(self, symbol: str, u: int) -> bool:
        """Return True if `u` did not advance by exactly one."""
        prev = self.last_u.get(symbol)
        self.last_u[symbol] = u
        if prev is None:
            return False
        self.deltas += 1
        if u == prev + 1:
            self.step_one += 1
            return False
        return True


class OkxBookTracker:
    """OKX books channel: each update's `prevSeqId` must equal the previous message's `seqId`."""

    def __init__(self) -> None:
        self.last_seq: dict[str, int] = {}

    def snapshot(self, inst: str, seq: int) -> None:
        self.last_seq[inst] = seq

    def update(self, inst: str, prev_seq: int, seq: int) -> bool:
        """Return True if a gap was detected."""
        last = self.last_seq.get(inst)
        self.last_seq[inst] = seq
        return last is not None and prev_seq != last
