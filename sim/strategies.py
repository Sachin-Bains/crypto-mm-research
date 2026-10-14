"""Strategies. Layer 4 adds the uncertainty-aware quoter; this is the naive baseline."""

from __future__ import annotations

from .engine import ASK, BID, Strategy


class JoinQuoter(Strategy):
    """Quote one fixed size at the best bid and best ask, always.

    Re-quotes whenever the best price moves, and stops quoting a side once the
    position limit is reached. It has no view on direction, so whatever it
    loses to informed flow shows up directly in the markouts.
    """

    def __init__(self, size: float = 0.01, max_pos: float = 0.05) -> None:
        self.size, self.max_pos = size, max_pos
        self.pos = 0.0
        self.live: dict[int, tuple[int, float] | None] = {BID: None, ASK: None}

    def on_book(self, t, bid, bid_qty, ask, ask_qty) -> None:
        for side, px in ((BID, bid), (ASK, ask)):
            allowed = (self.pos + self.size <= self.max_pos + 1e-12) if side == BID \
                else (self.pos - self.size >= -self.max_pos - 1e-12)
            cur = self.live[side]
            if cur is not None and (not allowed or cur[1] != px):
                self.api.cancel(cur[0])
                self.live[side] = cur = None
            if cur is None and allowed:
                self.live[side] = (self.api.place(side, px, self.size), px)

    def on_fill(self, t, oid, side, price, qty, done) -> None:
        self.pos += side * qty
        cur = self.live[side]
        if done and cur is not None and cur[0] == oid:
            self.live[side] = None

    def on_done(self, t, oid, reason) -> None:
        for side, cur in self.live.items():
            if cur is not None and cur[0] == oid:
                self.live[side] = None

    def on_invalid(self, t) -> None:
        self.live = {BID: None, ASK: None}
