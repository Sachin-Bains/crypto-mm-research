"""Strategies: the naive baseline (JoinQuoter) and the forecast- and uncertainty-aware quoter (Layer 4)."""

from __future__ import annotations

import math

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


class ForecastQuoter(Strategy):
    """Market maker that uses a short-horizon forecast and its uncertainty.

    At each best-price change it asks the signal what it knew at that moment:
    forecast f (bps), interval half-width w (bps) and rho = w / typical w.

      use_forecast  quote one side only when the forecast is strong relative to its
                    uncertainty: |f| > theta * w. A falling forecast pulls the bid,
                    a rising one pulls the ask.
      use_width     step back from the best price by depth_k ticks per unit of rho
                    above 1 (more uncertain than usual), up to max_depth ticks, and
                    withdraw both sides when rho > rho_max.
      inv_k         lean against inventory: both quotes move inv_k ticks per full
                    position limit, so a long position quotes lower and sells sooner.

    With everything off it behaves like JoinQuoter, but only quotes while a
    forecast exists, so all variants are compared over the same hours.
    """

    def __init__(self, signal, latency_ns: int, size: float = 0.01, max_pos: float = 0.05, tick: float = 0.1,
                 use_forecast: bool = True, use_width: bool = True, theta: float = 0.25,
                 depth_k: float = 5.0, max_depth: int = 20, rho_max: float = 2.5, inv_k: float = 2.0) -> None:
        self.signal, self.L = signal, latency_ns
        self.size, self.max_pos, self.tick = size, max_pos, tick
        self.use_forecast, self.use_width = use_forecast, use_width
        self.theta, self.depth_k, self.max_depth, self.rho_max, self.inv_k = theta, depth_k, max_depth, rho_max, inv_k
        self.pos = 0.0
        self.live: dict[int, tuple[int, float] | None] = {BID: None, ASK: None}
        self.calls = self.quoting = self.withdrawn = self.one_sided = 0

    def _round(self, px: float) -> float:
        return round(round(px / self.tick) * self.tick, 10)

    def desired(self, t, bid, ask) -> dict[int, float | None]:
        sig = self.signal.at(t - self.L)
        want: dict[int, float | None] = {BID: None, ASK: None}
        if sig is None:
            return want
        f, w, rho = sig
        if self.use_width and rho > self.rho_max:
            self.withdrawn += 1
            return want
        depth = 0
        if self.use_width:
            depth = min(self.max_depth, int(math.floor(self.depth_k * max(0.0, rho - 1.0))))
        shift = -round(self.inv_k * self.pos / self.max_pos) if self.max_pos > 0 else 0
        b = self._round(bid + (shift - depth) * self.tick)
        a = self._round(ask + (shift + depth) * self.tick)
        b = min(b, self._round(ask - self.tick))      # never cross: stay passive
        a = max(a, self._round(bid + self.tick))
        want[BID], want[ASK] = b, a
        if self.use_forecast and w > 0 and abs(f) > self.theta * w:
            want[BID if f < 0 else ASK] = None
            self.one_sided += 1
        if self.pos + self.size > self.max_pos + 1e-12:
            want[BID] = None
        if self.pos - self.size < -self.max_pos - 1e-12:
            want[ASK] = None
        return want

    def on_book(self, t, bid, bid_qty, ask, ask_qty) -> None:
        self.calls += 1
        want = self.desired(t, bid, ask)
        if want[BID] is not None or want[ASK] is not None:
            self.quoting += 1
        for side in (BID, ASK):
            cur, px = self.live[side], want[side]
            if cur is not None and cur[1] != px:
                self.api.cancel(cur[0])
                self.live[side] = cur = None
            if cur is None and px is not None:
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


VARIANTS = {
    "join": dict(use_forecast=False, use_width=False, inv_k=0.0),
    "forecast": dict(use_forecast=True, use_width=False),
    "width": dict(use_forecast=False, use_width=True),
    "full": dict(use_forecast=True, use_width=True),
}
