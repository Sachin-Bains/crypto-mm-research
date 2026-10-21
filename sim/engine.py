"""Event-driven fill simulator for passive (maker) orders against recorded L2 data.

Everything runs on the exchange's clock:
  * the strategy sees each book update `latency` after it happened at the exchange
  * its orders and cancels reach the exchange `latency` after it sends them
  * it learns of fills, cancels and rejects `latency` after they happen

Our orders are virtual: they never change the recorded book. We assume our
size is small relative to the queues, so we do not move the market.

Fill models for a resting order at price P:
  pessimistic  filled only when the market trades through P (or the book crosses it)
  queue        joins behind the visible size at P. Trades at P consume the queue
               ahead first. When the level shrinks, traded volume is subtracted
               first and only the remainder counts as cancels, which are split
               between ahead and behind in proportion to ahead^n : behind^n.
  optimistic   filled by any trade at P, as if at the front of the queue

Trade-through and crossed-book fills apply in all three models.

What the strategy sees: with real-time best bid and offer data (Binance
bookTicker), the strategy is driven by every change in the best prices, the
mid used for markouts comes from it, and a resting order is filled the moment
the opposite best price reaches it. Without it, both come from the 100 ms
depth stream, which hides any latency difference below 100 ms. Queue
positions are always tracked on the depth stream, which is the only source
of sizes below the top of the book.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from sortedcontainers import SortedDict

from .data import MarketData

BID, ASK = 1, -1
PESSIMISTIC, QUEUE, OPTIMISTIC = "pessimistic", "queue", "optimistic"
MODELS = (PESSIMISTIC, QUEUE, OPTIMISTIC)
EPS = 1e-12
INF = math.inf


@dataclass
class Order:
    oid: int
    side: int
    price: float
    qty: float
    sent_ns: int
    rem: float = 0.0
    arrive_ns: int = 0
    ahead: float = 0.0
    ahead0: float = 0.0
    status: str = "pending"  # pending, live, filled, cancelled, rejected


@dataclass
class Fill:
    t_ns: int
    oid: int
    side: int
    price: float
    qty: float
    mid: float
    ahead0: float
    wait_ns: int
    how: str = ""          # queue: our turn in the queue came; through: a trade went through our price; cross: the book crossed us


class Api:
    """What a strategy can do. Actions take effect at the exchange after the latency."""

    def __init__(self) -> None:
        self._actions: list = []
        self._ids = itertools.count(1)
        self.now_ns = 0

    def place(self, side: int, price: float, qty: float) -> int:
        oid = next(self._ids)
        self._actions.append(("place", Order(oid, side, price, qty, self.now_ns, rem=qty)))
        return oid

    def cancel(self, oid: int) -> None:
        self._actions.append(("cancel", oid))


class Strategy:
    """Base class. Override what you need."""

    def bind(self, api: Api) -> None:
        self.api = api

    def on_book(self, t_ns: int, bid: float, bid_qty: float, ask: float, ask_qty: float) -> None: ...
    def on_fill(self, t_ns: int, oid: int, side: int, price: float, qty: float, done: bool) -> None: ...
    def on_done(self, t_ns: int, oid: int, reason: str) -> None: ...
    def on_invalid(self, t_ns: int) -> None: ...


@dataclass
class Result:
    model: str
    latency_ns: int
    fills: list[Fill] = field(default_factory=list)
    mid_t: np.ndarray | None = None
    mid_v: np.ndarray | None = None
    position: float = 0.0
    cash: float = 0.0
    notional: float = 0.0
    orders_placed: int = 0
    orders_rejected: int = 0
    orders_cancelled_by_invalid: int = 0


class Engine:
    def __init__(self, md: MarketData, strategy: Strategy, latency_ns: int, model: str = QUEUE,
                 queue_power: float = 1.0, use_bbo: bool = True) -> None:
        if model not in MODELS:
            raise ValueError(model)
        self.md, self.strategy, self.L, self.model, self.n = md, strategy, int(latency_ns), model, queue_power
        self.use_bbo = use_bbo and md.has_bbo
        self.bbo: tuple | None = None   # latest (bid, bid_qty, ask, ask_qty) from the real-time feed
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()
        self.valid = False
        self.live: dict[int, Order] = {}
        self.traded: dict[tuple[int, float], float] = {}
        self.heap: list = []
        self.seq = itertools.count()
        self.api = Api()
        strategy.bind(self.api)
        self.res = Result(model, self.L)
        self._mid_t: list[int] = []
        self._mid_v: list[float] = []
        self._last_top = None
        self.now = 0

    # Book helpers -----------------------------------------------------------
    def best_bid(self) -> float:
        if self.bbo is not None:
            return self.bbo[0]
        return self.bids.peekitem(-1)[0] if self.bids else -INF

    def best_ask(self) -> float:
        if self.bbo is not None:
            return self.bbo[2]
        return self.asks.peekitem(0)[0] if self.asks else INF

    def mid(self) -> float:
        bb, ba = self.best_bid(), self.best_ask()
        if bb > -INF and ba < INF:
            return 0.5 * (bb + ba)
        return math.nan

    def level_qty(self, side: int, price: float) -> float:
        return (self.bids if side == BID else self.asks).get(price, 0.0)

    # Scheduling -------------------------------------------------------------
    def schedule(self, t: int, kind: str, payload) -> None:
        heapq.heappush(self.heap, (t, next(self.seq), kind, payload))

    def _flush_actions(self, t: int) -> None:
        for kind, x in self.api._actions:
            if kind == "place":
                x.sent_ns = t
                self.res.orders_placed += 1
                self.schedule(t + self.L, "place", x)
            else:
                self.schedule(t + self.L, "cancel", x)
        self.api._actions.clear()

    def _call(self, t: int, fn, *args) -> None:
        self.api.now_ns = t
        fn(t, *args)
        self._flush_actions(t)

    # Fills ------------------------------------------------------------------
    def _fill(self, o: Order, x: float, t: int, how: str = "") -> None:
        if x <= EPS:
            return
        x = min(x, o.rem)
        o.rem -= x
        done = o.rem <= EPS
        self.res.fills.append(Fill(t, o.oid, o.side, o.price, x, self.mid(), o.ahead0, t - o.arrive_ns, how))
        self.res.position += o.side * x
        self.res.cash -= o.side * x * o.price
        self.res.notional += x * o.price
        if done:
            o.status = "filled"
            self.live.pop(o.oid, None)
        self.schedule(t + self.L, "report_fill", (o.oid, o.side, o.price, x, done))

    def _check_cross(self, t: int, bb: float, ba: float) -> None:
        for o in list(self.live.values()):
            if (o.side == BID and o.price >= ba) or (o.side == ASK and o.price <= bb):
                self._fill(o, o.rem, t, "cross")

    # Market events ----------------------------------------------------------
    def _on_trade(self, i: int) -> None:
        if not self.valid:
            return
        md = self.md
        t, p, v, s = int(md.t_exch[i]), float(md.t_price[i]), float(md.t_qty[i]), int(md.t_side[i])
        passive = -s
        key = (passive, p)
        self.traded[key] = self.traded.get(key, 0.0) + v
        for o in list(self.live.values()):
            if o.side != passive:
                continue
            through = p < o.price if passive == BID else p > o.price
            if through:
                self._fill(o, o.rem, t, "through")
            elif p == o.price:
                if self.model == OPTIMISTIC:
                    self._fill(o, v, t, "queue")
                elif self.model == QUEUE:
                    if v > o.ahead + EPS:
                        x = v - o.ahead
                        o.ahead = 0.0
                        self._fill(o, x, t, "queue")
                    else:
                        o.ahead -= v

    def _cancel_all(self, t: int, reason: str) -> None:
        for o in list(self.live.values()):
            o.status = "cancelled"
            self.live.pop(o.oid)
            self.res.orders_cancelled_by_invalid += 1
            self.schedule(t + self.L, "report_done", (o.oid, reason))

    def _on_depth(self, g: int) -> None:
        md = self.md
        k0, k1 = int(md.g_start[g]), int(md.g_end[g])
        t = int(md.d_exch[k0])
        kind = md.d_kind[k0]
        if kind == 2:
            self.valid = False
            self.bbo = None
            self.traded.clear()
            self._cancel_all(t, "invalid")
            self._last_top = None
            self.schedule(t + self.L, "invalid", None)
            return
        if kind == 1:
            self.bids.clear()
            self.asks.clear()
            k0 += 1
            self.valid = True

        # remember level sizes before the update for levels where we rest
        touched: dict[tuple[int, float], float] = {}
        if self.live:
            ours = {(o.side, o.price) for o in self.live.values()}
        else:
            ours = ()
        sides, prices, qtys = md.d_side, md.d_price, md.d_qty
        bids, asks = self.bids, self.asks
        for k in range(k0, k1):
            side, px, q = sides[k], float(prices[k]), float(qtys[k])
            book = bids if side == BID else asks
            if ours and (side, px) in ours and (side, px) not in touched:
                touched[(side, px)] = book.get(px, 0.0)
            if q == 0.0:
                book.pop(px, None)
            else:
                book[px] = q

        if self.model == QUEUE:
            for (side, px), q_old in touched.items():
                q_new = self.level_qty(side, px)
                traded = self.traded.get((side, px), 0.0)
                cancels = max(0.0, (q_old - q_new) - traded)
                for o in self.live.values():
                    if o.side != side or o.price != px:
                        continue
                    if cancels > EPS and o.ahead > EPS:
                        behind = max(0.0, q_old - traded - o.ahead)
                        a, b = o.ahead ** self.n, behind ** self.n
                        o.ahead -= cancels * (a / (a + b) if a + b > 0 else 1.0)
                    o.ahead = max(0.0, min(o.ahead, q_new))
        elif kind == 1:
            for o in self.live.values():
                o.ahead = max(0.0, min(o.ahead, self.level_qty(o.side, o.price)))
        self.traded.clear()

        if not self.valid:
            return
        if not (bids and asks):
            return
        dbb, dba = bids.peekitem(-1)[0], asks.peekitem(0)[0]
        self._check_cross(t, dbb, dba)
        if self.use_bbo:
            return
        self._mid_t.append(t)
        self._mid_v.append(0.5 * (dbb + dba))
        top = (dbb, bids[dbb], dba, asks[dba])
        if top != self._last_top:
            self._last_top = top
            self.schedule(t + self.L, "book", top)

    def _on_bbo(self, i: int) -> None:
        if not self.valid:
            return
        md = self.md
        t = int(md.b_exch[i])
        top = (float(md.b_bid[i]), float(md.b_bid_qty[i]), float(md.b_ask[i]), float(md.b_ask_qty[i]))
        self.bbo = top
        self._check_cross(t, top[0], top[2])
        self._mid_t.append(t)
        self._mid_v.append(0.5 * (top[0] + top[2]))
        self.schedule(t + self.L, "book", top)

    # Our order events -------------------------------------------------------
    def _on_place(self, t: int, o: Order) -> None:
        if not self.valid or (o.side == BID and o.price >= self.best_ask()) or (o.side == ASK and o.price <= self.best_bid()):
            o.status = "rejected"   # post-only: an order that would trade immediately is rejected
            self.res.orders_rejected += 1
            self.schedule(t + self.L, "report_done", (o.oid, "rejected"))
            return
        o.status, o.arrive_ns = "live", t
        ahead = self.level_qty(o.side, o.price)
        if self.bbo is not None:
            # the real-time feed has the freshest size at the best price
            if o.side == BID and o.price == self.bbo[0]:
                ahead = self.bbo[1]
            elif o.side == ASK and o.price == self.bbo[2]:
                ahead = self.bbo[3]
        o.ahead = o.ahead0 = ahead
        self.live[o.oid] = o

    def _on_cancel(self, t: int, oid: int) -> None:
        o = self.live.pop(oid, None)
        if o is not None:
            o.status = "cancelled"
            self.schedule(t + self.L, "report_done", (oid, "cancelled"))

    # Main loop --------------------------------------------------------------
    def run(self) -> Result:
        md = self.md
        G, T = len(md.g_start), len(md.t_exch)
        B = len(md.b_exch) if self.use_bbo else 0
        gi = ti = bi = 0
        d_first = md.d_exch[md.g_start] if G else np.zeros(0, dtype=np.int64)
        last_market = max(int(d_first[-1]) if G else 0, int(md.t_exch[-1]) if T else 0,
                          int(md.b_exch[B - 1]) if B else 0)
        strat = self.strategy
        while True:
            t_d = d_first[gi] if gi < G else INF
            t_t = md.t_exch[ti] if ti < T else INF
            t_b = md.b_exch[bi] if bi < B else INF
            t_i = self.heap[0][0] if self.heap else INF
            if t_d == INF and t_t == INF and t_b == INF and t_i == INF:
                break
            # same-time order: trades, then depth, then best bid/offer, then our own events
            if t_t <= t_d and t_t <= t_b and t_t <= t_i:
                self.now = int(t_t)
                self._on_trade(ti)
                ti += 1
            elif t_d <= t_b and t_d <= t_i:
                self.now = int(t_d)
                self._on_depth(gi)
                gi += 1
            elif t_b <= t_i:
                self.now = int(t_b)
                self._on_bbo(bi)
                bi += 1
            else:
                if gi >= G and ti >= T and bi >= B and t_i > last_market:
                    break  # data finished; events after the last market time cannot be simulated
                t, _, kind, x = heapq.heappop(self.heap)
                self.now = t
                if kind == "book":
                    self._call(t, strat.on_book, *x)
                elif kind == "place":
                    self._on_place(t, x)
                elif kind == "cancel":
                    self._on_cancel(t, x)
                elif kind == "report_fill":
                    self._call(t, strat.on_fill, *x)
                elif kind == "report_done":
                    self._call(t, strat.on_done, *x)
                elif kind == "invalid":
                    self._call(t, strat.on_invalid)
        self.res.mid_t = np.asarray(self._mid_t, dtype=np.int64)
        self.res.mid_v = np.asarray(self._mid_v, dtype=np.float64)
        return self.res
