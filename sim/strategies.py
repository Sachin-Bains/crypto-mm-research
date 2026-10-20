"""Strategies: the naive baseline (JoinQuoter) and the forecast- and uncertainty-aware quoter (Layer 4)."""

from __future__ import annotations

import math
import random

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
      pull_at       alternative inventory control: at pull_at of the limit, stop quoting
                    the side that would add to the position, and keep the other at the touch.

    In a market whose spread is one tick (BTCUSDT perpetual), a quote one tick behind the
    best price is only filled when the price moves through it, which is the most adversely
    selected fill there is. So "step back" rules (the lean, the width depth) are expected to
    hurt there, and "pull" rules are the alternative.

      vol_max       control: withdraw both sides when realised volatility (rv_30s) is above
                    vol_max times its typical level. Same structure as the width rule, but from
                    volatility alone, with no forecast and no conformal interval.
      random_off    control: at each best-price change, withdraw both sides with this
                    probability, using no information. A strategy that avoids losing fills
                    must beat this at the same fill count, or it is just trading less.

    With everything off it behaves like JoinQuoter, but only quotes while a
    forecast exists, so all variants are compared over the same hours.
    """

    def __init__(self, signal, latency_ns: int, size: float = 0.01, max_pos: float = 0.05, tick: float = 0.1,
                 use_forecast: bool = True, use_width: bool = True, theta: float = 0.25,
                 depth_k: float = 5.0, max_depth: int = 20, rho_max: float = 2.5, inv_k: float = 2.0,
                 pull_at: float | None = None, random_off: float = 0.0, seed: int = 0,
                 vol_max: float | None = None) -> None:
        self.signal, self.L = signal, latency_ns
        self.size, self.max_pos, self.tick = size, max_pos, tick
        self.use_forecast, self.use_width = use_forecast, use_width
        self.theta, self.depth_k, self.max_depth, self.rho_max, self.inv_k = theta, depth_k, max_depth, rho_max, inv_k
        self.pull_at = pull_at
        self.random_off = random_off
        self.vol_max = vol_max
        self._rng = random.Random(seed)
        self.pos = 0.0
        self.live: dict[int, tuple[int, float] | None] = {BID: None, ASK: None}
        self.calls = self.quoting = self.withdrawn = self.one_sided = self.no_signal = 0
        self.requotes = 0   # cancels of a live quote to move or pull it: each one gives up queue position

    def _round(self, px: float) -> float:
        return round(round(px / self.tick) * self.tick, 10)

    def desired(self, t, bid, ask) -> dict[int, float | None]:
        sig = self.signal.at(t - self.L)
        want: dict[int, float | None] = {BID: None, ASK: None}
        if sig is None:
            self.no_signal += 1
            return want
        f, w, rho = sig
        if self.vol_max is not None:
            vr = self.signal.vol_ratio(t - self.L)
            if vr is None or vr > self.vol_max:
                self.withdrawn += 1   # control: withdraw when recent volatility is high, no forecast
                return want
        if self.random_off > 0 and self._rng.random() < self.random_off:
            self.withdrawn += 1       # control: withdraw at random, with no information
            return want
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
        if self.pull_at is not None:
            # inventory control by pulling, not stepping back: once the position reaches
            # pull_at of the limit, stop quoting the side that would add to it
            if self.pos >= self.pull_at * self.max_pos - 1e-12:
                want[BID] = None
            if self.pos <= -self.pull_at * self.max_pos + 1e-12:
                want[ASK] = None
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
                self.requotes += 1
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
    # ablations: the variants above also differ from join by the inventory skew (inv_k = 2),
    # so these separate the effect of each component
    "join_skew": dict(use_forecast=False, use_width=False),
    "forecast_noskew": dict(use_forecast=True, use_width=False, inv_k=0.0),
    "width_noskew": dict(use_forecast=False, use_width=True, inv_k=0.0),
    # pull instead of stepping back
    "join_pull": dict(use_forecast=False, use_width=False, inv_k=0.0, pull_at=0.4),
    "forecast_pull": dict(use_forecast=True, use_width=False, inv_k=0.0, pull_at=0.4),
    "width_pull": dict(use_forecast=False, use_width=True, inv_k=0.0, depth_k=0.0, rho_max=1.5),
    "full_pull": dict(use_forecast=True, use_width=True, inv_k=0.0, depth_k=0.0, rho_max=1.5, pull_at=0.4),
    # controls: join with random withdrawals (rate set by --random-off) and the pull rule
    "random_off": dict(use_forecast=False, use_width=False, inv_k=0.0),
    "random_off_pull": dict(use_forecast=False, use_width=False, inv_k=0.0, pull_at=0.4),
    # control: a plain volatility filter with the same inventory rule (threshold set by --vol-max or --match)
    "vol_pull": dict(use_forecast=False, use_width=False, inv_k=0.0, pull_at=0.4, vol_max=1.5),
}
DEFAULT_SET = ("join", "forecast", "width", "full")


class ConformalBudgetQuoter(ForecastQuoter):
    """Version 2: one rule and one dial, an adverse-selection budget.

    Score per side = the interval's worst-case adverse move against that side, in bp:
        bid: w - f    (how far the price could fall, at the interval's coverage)
        ask: w + f    (how far it could rise)
    It combines the forecast f and its uncertainty w with no hand-set weights.
    A side is quoted only while its score is at or below a threshold lam.

    lam is not chosen by hand. It is tuned online so that the realised adverse
    selection of the strategy's own fills stays at the budget B (bp per unit traded):
    each fill's 5 s markout loss is settled once its outcome is known (5 s later,
    from the strategy's own view of the book), and
        lam <- lam - gamma * (loss - B)
    so fills worse than the budget make the strategy more selective, and better
    ones let it quote more. This is the update of adaptive conformal inference
    applied to a loss instead of a miss indicator (online conformal risk control).
    With losses clipped to [-C, C] and lam staying inside [lam_lo, lam_hi], the
    average settled loss over N fills is within (lam_hi - lam_lo) / (gamma * N) of B,
    whatever the market does. If lam sits on a bound that guarantee does not hold,
    and the run reports how often it did.

    Inventory is controlled as in full_pull: no adding to a position past pull_at of the limit.
    """

    def __init__(self, signal, latency_ns: int, budget_bp: float = 0.3, gamma: float = 0.02,
                 lam0: float = 3.0, lam_lo: float = 0.5, lam_hi: float = 10.0, horizon_s: float = 5.0,
                 loss_clip_bp: float = 20.0, size: float = 0.01, max_pos: float = 0.05, tick: float = 0.1,
                 pull_at: float | None = 0.4) -> None:
        super().__init__(signal, latency_ns, size=size, max_pos=max_pos, tick=tick,
                         use_forecast=False, use_width=False, inv_k=0.0, pull_at=pull_at)
        self.B, self.gamma, self.lam = budget_bp, gamma, lam0
        self.lam_lo, self.lam_hi, self.h = lam_lo, lam_hi, int(horizon_s * 1e9)
        self.C = loss_clip_bp
        self.pending: list[tuple[int, int, float, float]] = []   # (settle_at, side, price, qty)
        self.settled = 0
        self.loss_sum = 0.0          # clipped, qty-weighted (in units of quote size)
        self.weight_sum = 0.0
        self.at_bound = 0
        self.trace: list[tuple[int, float, float]] = []   # (t, lam, running mean loss), every 50 fills

    def _settle(self, t: int, mid: float) -> None:
        keep = []
        for settle_at, side, price, qty in self.pending:
            if t < settle_at:
                keep.append((settle_at, side, price, qty))
                continue
            loss = -side * (mid - price) / price * 1e4            # positive = moved against us
            loss = min(self.C, max(-self.C, loss))
            wgt = qty / self.size
            self.lam -= self.gamma * wgt * (loss - self.B)
            if self.lam <= self.lam_lo or self.lam >= self.lam_hi:
                self.at_bound += 1
            self.lam = min(self.lam_hi, max(self.lam_lo, self.lam))
            self.loss_sum += wgt * loss
            self.weight_sum += wgt
            self.settled += 1
            if self.settled % 50 == 0:
                self.trace.append((t, round(self.lam, 4), round(self.loss_sum / self.weight_sum, 4)))
        self.pending = keep

    def desired(self, t, bid, ask) -> dict[int, float | None]:
        if self.pending and math.isfinite(bid) and math.isfinite(ask):
            self._settle(t, 0.5 * (bid + ask))
        want: dict[int, float | None] = {BID: None, ASK: None}
        sig = self.signal.at(t - self.L)
        if sig is None:
            self.no_signal += 1
            return want
        f, w, _ = sig
        if self.w_ok(w):
            if w - f <= self.lam:
                want[BID] = self._round(bid)
            if w + f <= self.lam:
                want[ASK] = self._round(ask)
        if (want[BID] is None) != (want[ASK] is None):
            self.one_sided += 1
        if want[BID] is None and want[ASK] is None:
            self.withdrawn += 1
        if self.pull_at is not None:
            if self.pos >= self.pull_at * self.max_pos - 1e-12:
                want[BID] = None
            if self.pos <= -self.pull_at * self.max_pos + 1e-12:
                want[ASK] = None
        if self.pos + self.size > self.max_pos + 1e-12:
            want[BID] = None
        if self.pos - self.size < -self.max_pos - 1e-12:
            want[ASK] = None
        return want

    @staticmethod
    def w_ok(w: float) -> bool:
        return math.isfinite(w) and w > 0

    def on_fill(self, t, oid, side, price, qty, done) -> None:
        self.pending.append((t + self.h, side, price, qty))
        super().on_fill(t, oid, side, price, qty, done)

    def report(self) -> dict:
        n = max(self.weight_sum, 1e-12)
        return {"budget_bp": self.B, "gamma": self.gamma, "lam_final": round(self.lam, 4),
                "settled_fills": self.settled, "realised_loss_bp": round(self.loss_sum / n, 4),
                "share_updates_at_bound": round(self.at_bound / max(self.settled, 1), 4),
                "guarantee_slack_bp": round((self.lam_hi - self.lam_lo) / (self.gamma * n), 4),
                "trace": self.trace}
