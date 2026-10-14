"""Hand-worked tests for the fill simulator.

Each scenario is small enough to check on paper. Times are in milliseconds
on the exchange clock; latency is 10 ms each way unless stated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.analysis import summarise  # noqa: E402
from sim.data import MarketData  # noqa: E402
from sim.engine import ASK, BID, OPTIMISTIC, PESSIMISTIC, QUEUE, Engine, Strategy  # noqa: E402

MS = 1_000_000


class Script:
    """Builds book messages and trades in time order."""

    def __init__(self) -> None:
        self.d = []      # (exch, msg, kind, side, price, qty)
        self.t = []      # (exch, price, qty, side)
        self.msg = 0

    def reset(self, ms, bids, asks):
        self.msg += 1
        self.d.append((ms * MS, self.msg, 1, 0, 0.0, 0.0))
        for p, q in bids:
            self.d.append((ms * MS, self.msg, 0, BID, p, q))
        for p, q in asks:
            self.d.append((ms * MS, self.msg, 0, ASK, p, q))
        return self

    def update(self, ms, *levels):
        self.msg += 1
        for side, p, q in levels:
            self.d.append((ms * MS, self.msg, 0, side, p, q))
        return self

    def invalid(self, ms):
        self.msg += 1
        self.d.append((ms * MS, self.msg, 2, 0, 0.0, 0.0))
        return self

    def trade(self, ms, price, qty, aggressor):
        self.t.append((ms * MS, price, qty, aggressor))
        return self

    def data(self) -> MarketData:
        d = np.array(self.d, dtype=object) if self.d else np.zeros((0, 6))
        t = sorted(self.t)
        col = lambda rows, i: [r[i] for r in rows]  # noqa: E731
        return MarketData.from_arrays(col(self.d, 0), col(self.d, 1), col(self.d, 2), col(self.d, 3),
                                      col(self.d, 4), col(self.d, 5),
                                      col(t, 0), col(t, 1), col(t, 2), col(t, 3))


class OneOrder(Strategy):
    """Places one order on the first book it sees, optionally cancels it later."""

    def __init__(self, side, price, qty=1.0, cancel_at_ms=None):
        self.side, self.price, self.qty, self.cancel_at = side, price, qty, cancel_at_ms
        self.oid = None
        self.fills, self.done = [], []

    def on_book(self, t, bid, bq, ask, aq):
        if self.oid is None:
            self.oid = self.api.place(self.side, self.price, self.qty)
        elif self.cancel_at is not None and t >= self.cancel_at * MS:
            self.api.cancel(self.oid)
            self.cancel_at = None

    def on_fill(self, t, oid, side, price, qty, done):
        self.fills.append((t // MS, qty, done))

    def on_done(self, t, oid, reason):
        self.done.append((t // MS, reason))


def run(script, strat, model, latency_ms=10, n=1.0):
    return Engine(script.data(), strat, latency_ms * MS, model, n).run()


def base():
    # bid 100.0 x 5, ask 100.1 x 5. Order sent at 10 ms (sees book at 0+10), arrives at 20 ms.
    return Script().reset(0, [(100.0, 5.0), (99.9, 3.0)], [(100.1, 5.0), (100.2, 3.0)])


# --------------------------------------------------------------------------- queue arithmetic

def test_queue_trade_then_cancel_then_trade():
    s = base()
    s.trade(30, 100.0, 2.0, -1)                 # 2 traded at our level: ahead 5 -> 3
    s.update(40, (BID, 100.0, 2.5))             # level 5 -> 2.5: 2 was trades, 0.5 cancels.
    #   behind us = 5 - 2 - 3 = 0, so all 0.5 cancels were ahead: ahead 3 -> 2.5
    s.trade(50, 100.0, 3.0, -1)                 # 3 traded: 2.5 clears what is ahead, 0.5 fills us
    s.update(60, (BID, 100.0, 1.0))             # keep the book moving
    s.trade(70, 100.0, 1.0, -1)                 # 1 more: fills our remaining 0.5
    s.update(80, (BID, 100.0, 0.5))
    strat = OneOrder(BID, 100.0, qty=1.0)
    res = run(s, strat, QUEUE)
    qtys = [round(f.qty, 9) for f in res.fills]
    assert qtys == [0.5, 0.5]
    assert res.fills[0].ahead0 == 5.0
    assert res.position == pytest.approx(1.0)
    assert res.cash == pytest.approx(-100.0)


def test_trades_are_not_double_counted_as_cancels():
    """The book update after a trade includes that trade. If it were counted again as
    cancels, the order would jump the queue and fill on the next small trade."""
    s = base()
    s.trade(30, 100.0, 4.0, -1)                 # ahead 5 -> 1
    s.update(40, (BID, 100.0, 1.0))             # 5 -> 1 is exactly the 4 traded: no cancels
    s.trade(50, 100.0, 0.5, -1)                 # ahead 1 -> 0.5, still no fill
    s.update(60, (BID, 100.0, 0.5))
    res = run(s, OneOrder(BID, 100.0), QUEUE)
    assert res.fills == []


def test_cancels_split_by_queue_position():
    s = base()
    s.update(30, (BID, 100.0, 9.0))             # 4 join behind us: ahead 5, behind 4
    s.update(40, (BID, 100.0, 6.0))             # 3 cancelled: ahead share 5/9 -> ahead 5 - 3*5/9
    s.trade(50, 100.0, 5.0 - 3 * 5 / 9 + 0.25, -1)  # just enough to clear ahead plus 0.25
    s.update(60, (BID, 100.0, 1.0))
    res = run(s, OneOrder(BID, 100.0), QUEUE)
    assert len(res.fills) == 1 and res.fills[0].qty == pytest.approx(0.25)


# --------------------------------------------------------------------------- the three models

def test_models_rank_pessimistic_queue_optimistic():
    s = base()
    s.trade(30, 100.0, 1.0, -1)                 # small trade at our price
    s.update(40, (BID, 100.0, 4.0))
    s.trade(50, 99.9, 0.5, -1)                  # trades through our price
    s.update(60, (BID, 100.0, 0.0), (BID, 99.9, 2.5))
    fills = {}
    for model in (PESSIMISTIC, QUEUE, OPTIMISTIC):
        res = run(s, OneOrder(BID, 100.0), model)
        fills[model] = [(f.t_ns // MS, f.qty) for f in res.fills]
    assert fills[OPTIMISTIC] == [(30, 1.0)]           # first trade at our price fills it all
    assert fills[QUEUE] == [(50, 1.0)]                # stuck behind 5 until the price trades through
    assert fills[PESSIMISTIC] == [(50, 1.0)]


def test_crossed_book_fills_all_models():
    s = base()
    s.update(30, (ASK, 100.1, 0.0), (ASK, 100.0, 2.0), (BID, 100.0, 0.0), (BID, 99.9, 3.0))
    for model in (PESSIMISTIC, QUEUE, OPTIMISTIC):
        res = run(s, OneOrder(BID, 100.0), model)
        assert len(res.fills) == 1 and res.fills[0].t_ns == 30 * MS


# --------------------------------------------------------------------------- latency

def test_order_joins_queue_as_it_is_when_it_arrives():
    s = base()
    s.update(15, (BID, 100.0, 8.0))             # queue grows before our order lands at 20 ms
    s.trade(30, 100.0, 8.5, -1)
    s.update(40, (BID, 100.0, 0.5))
    res = run(s, OneOrder(BID, 100.0), QUEUE)
    assert res.fills[0].ahead0 == 8.0 and res.fills[0].qty == pytest.approx(0.5)


class CancelOnSignal(OneOrder):
    """Cancels as soon as it sees the 50 ms book update (ask size 5.5)."""

    def on_book(self, t, bid, bq, ask, aq):
        if self.oid is None:
            self.oid = self.api.place(self.side, self.price, self.qty)
        elif aq == 5.5 and self.cancel_at is not None:
            self.api.cancel(self.oid)
            self.cancel_at = None


def test_cancel_loses_race_to_trade_through():
    s = base()
    for ms in (30, 40, 50):
        s.update(ms, (ASK, 100.1, 5.0 + ms / 100))    # top-of-book changes, so the strategy is called
    s.trade(65, 99.8, 1.0, -1)                        # price falls through our bid at 65 ms
    s.update(70, (BID, 100.0, 0.0), (BID, 99.9, 0.0))
    # 10 ms: sees the 50 ms update at 60, cancel lands at 70, after the 65 ms trade: filled
    slow = CancelOnSignal(BID, 100.0, cancel_at_ms=0)
    res = run(s, slow, QUEUE, latency_ms=10)
    assert len(res.fills) == 1 and res.fills[0].t_ns == 65 * MS
    # 1 ms: sees it at 51, cancel lands at 52, before the trade: not filled
    fast = CancelOnSignal(BID, 100.0, cancel_at_ms=0)
    res_fast = run(s, fast, QUEUE, latency_ms=1)
    assert res_fast.fills == [] and fast.done == [(53, "cancelled")]


def test_post_only_rejects_crossing_order():
    s = base()
    s.update(30, (BID, 99.9, 3.0))
    strat = OneOrder(BID, 100.1)
    res = run(s, strat, QUEUE)
    assert res.orders_rejected == 1 and strat.done == [(30, "rejected")]


def test_invalid_book_pulls_orders_and_blocks_fills():
    s = base()
    s.invalid(30)
    s.trade(40, 99.0, 5.0, -1)                  # would trade through, but the book is invalid
    s.reset(50, [(99.0, 1.0)], [(99.1, 1.0)])
    res = run(s, OneOrder(BID, 100.0), QUEUE)
    assert res.fills == [] and res.orders_cancelled_by_invalid == 1


# --------------------------------------------------------------------------- accounting

def test_pnl_split_round_trip():
    s = base()
    s.trade(30, 99.9, 1.0, -1)                  # our bid at 100.0 fills (trade through), mid 100.05
    s.update(40, (BID, 100.0, 0.0), (BID, 99.95, 1.0))
    for ms in range(1000, 33000, 1000):          # mid moves down 0.1 over the next 30 s
        s.update(ms, (ASK, 100.1, 0.0), (ASK, 100.05, 2.0), (BID, 99.95, 0.0), (BID, 99.9, 1.0)) if ms == 1000 \
            else s.update(ms, (BID, 99.9, 1.0 + ms / 1e6))
    res = run(s, OneOrder(BID, 100.0), QUEUE)
    out = summarise(res)
    assert out["fills"] == 1
    assert out["spread_captured_usd"] == pytest.approx(0.05)   # mid 100.05 - price 100.0
    # mid after 1 s is (99.9 + 100.05)/2 = 99.975 -> adverse = -0.075 per unit on 100 notional
    assert out["adverse_1s_bps"] == pytest.approx(-0.075 / 100.0 * 1e4)
    assert out["markout_1s_bps"] == pytest.approx(-0.025 / 100.0 * 1e4)
    assert out["pnl_gross_usd"] == pytest.approx(out["spread_captured_usd"] + out["inventory_usd"])
