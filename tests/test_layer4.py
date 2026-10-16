"""Tests for Layer 4: signal alignment, quoting rules, and an end-to-end check with a perfect forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sim.analysis import paired_vs, summarise  # noqa: E402
from sim.engine import ASK, BID, QUEUE, Api, Engine  # noqa: E402
from sim.signal import Signal  # noqa: E402
from sim.strategies import VARIANTS, ForecastQuoter  # noqa: E402
from test_sim import Script  # noqa: E402

MS = 1_000_000


# --------------------------------------------------------------------------- signal alignment

def test_signal_uses_only_what_had_been_received():
    # Binance updates at exchange times 0, 100, 200 ms reached us at 150, 250, 350 ms
    bx = np.array([0, 100, 200]) * MS
    br = np.array([150, 250, 350]) * MS
    # forecasts made at receive times 140, 240, 340 ms
    sig = Signal(np.array([140, 240, 340]) * MS, np.array([1.0, 2.0, 3.0]), np.ones(3), bx, br)
    assert sig.at(99 * MS)[0] == 1.0     # has seen the 0 ms update, received at 150: latest forecast is 140
    assert sig.at(100 * MS)[0] == 2.0    # has seen the 100 ms update, received at 250: forecast 240
    assert sig.at(-1) is None            # has seen nothing yet


def test_signal_goes_stale_and_rho_is_causal():
    bx = np.array([0, 10_000]) * MS
    sig = Signal(np.array([0, 1]) * MS, np.array([0.5, 0.5]), np.array([1.0, 4.0]), bx, bx)
    assert sig.at(10_000 * MS) is None   # newest forecast is 10 s old: stale
    f, w, rho = Signal(np.array([0, 100]) * MS, np.zeros(2), np.array([1.0, 4.0]), bx, bx).at(0)
    assert rho == 1.0                    # at t = 0 only the first width is known


# --------------------------------------------------------------------------- quoting rules

class Fixed:
    def __init__(self, f=0.0, w=1.0, rho=1.0):
        self.v = (f, w, rho)

    def at(self, t):
        return self.v


def quotes(f=0.0, w=1.0, rho=1.0, pos=0.0, **kw):
    q = ForecastQuoter(Fixed(f, w, rho), 0, tick=0.1, max_pos=0.05, **kw)
    q.bind(Api())
    q.pos = pos
    return q.desired(0, 100.0, 100.1)


def test_neutral_signal_joins_the_best_prices():
    assert quotes() == {BID: 100.0, ASK: 100.1}


def test_strong_forecast_pulls_the_side_that_would_be_picked_off():
    assert quotes(f=-0.5, w=1.0) == {BID: None, ASK: 100.1}     # price expected to fall: no bid
    assert quotes(f=+0.5, w=1.0) == {BID: 100.0, ASK: None}
    assert quotes(f=-0.2, w=1.0) == {BID: 100.0, ASK: 100.1}    # weak relative to uncertainty: both sides


def test_wide_uncertainty_steps_back_then_withdraws():
    assert quotes(rho=2.0) == {BID: 99.5, ASK: 100.6}           # 5 ticks per unit of rho above 1
    assert quotes(rho=3.0) == {BID: None, ASK: None}            # beyond rho_max = 2.5


def test_inventory_skew_and_never_crossing():
    # long: wants both quotes 1 tick lower, but an ask at 100.0 would hit the bid, so it stays at 100.1
    assert quotes(pos=0.025, use_forecast=False, use_width=False) == {BID: 99.9, ASK: 100.1}
    # short: wants to quote 2 ticks higher, but the bid may not reach the ask
    assert quotes(pos=-0.05 + 0.01, use_forecast=False, use_width=False) == {BID: 100.0, ASK: 100.3}


def test_pull_rule_drops_the_adding_side_and_keeps_the_other_at_the_touch():
    kw = dict(use_forecast=False, use_width=False, inv_k=0.0, pull_at=0.4)
    assert quotes(pos=0.01, **kw) == {BID: 100.0, ASK: 100.1}     # below 40% of the limit: both sides
    assert quotes(pos=0.02, **kw) == {BID: None, ASK: 100.1}      # at 40% long: no more buying
    assert quotes(pos=-0.02, **kw) == {BID: 100.0, ASK: None}


def test_paired_comparison_by_hour():
    base = {0: 10.0, 1: -5.0, 2: 0.0}
    better = {0: 12.0, 1: -2.0, 2: 1.0}               # +2, +3, +1 per hour
    d, t, n = paired_vs(base, better)
    assert n == 3 and d == 2.0 and t > 3
    d, t, n = paired_vs(base, {0: 10.0, 1: -5.0})     # hour 2 without fills counts as 0
    assert d == 0.0 and n == 3


def test_random_control_withdraws_at_the_set_rate():
    q = ForecastQuoter(Fixed(), 0, tick=0.1, use_forecast=False, use_width=False, inv_k=0.0, random_off=0.3)
    q.bind(Api())
    off = sum(q.desired(0, 100.0, 100.1) == {BID: None, ASK: None} for _ in range(10_000))
    assert abs(off / 10_000 - 0.3) < 0.02


def test_position_limit_stops_adding():
    assert quotes(pos=0.05, use_forecast=False, use_width=False)[BID] is None


# --------------------------------------------------------------------------- end to end

def informed_market(steps=200, seed=0):
    """The price moves one tick every second. 50 ms before each move, an informed trader sweeps
    the side about to be repriced. The perfect forecast knows the next move."""
    rng = np.random.default_rng(seed)
    s = Script()
    p = 100.0
    s.reset(0, [(p, 5.0), (round(p - 0.1, 1), 5.0)], [(round(p + 0.1, 1), 5.0), (round(p + 0.2, 1), 5.0)])
    s.bbo(0, p, 5.0, round(p + 0.1, 1), 5.0)
    times, preds = [0], [0.0]
    for k in range(1, steps + 1):
        T = k * 1000
        d = 1 if rng.random() < 0.5 else -1
        times.append(T - 1000)                   # forecast of this move is available as the last one happens
        preds.append(d * 1.0)
        if d < 0:
            s.trade(T - 50, round(p - 0.1, 1), 6.0, -1)      # sells through the bid at p
        else:
            s.trade(T - 50, round(p + 0.2, 1), 6.0, +1)      # buys through the ask at p + 0.1
        p = round(p + d * 0.1, 1)
        s.update(T, (BID, round(p + 0.1, 1), 0.0), (BID, p, 5.0), (BID, round(p - 0.1, 1), 5.0),
                 (ASK, p, 0.0), (ASK, round(p + 0.1, 1), 5.0), (ASK, round(p + 0.2, 1), 5.0))
        s.bbo(T, p, 5.0, round(p + 0.1, 1), 5.0)
    md = s.data()
    sig = Signal(np.array(times) * MS, np.array(preds), np.full(len(times), 1.0), md.b_exch, md.b_exch)
    return md, sig


def test_perfect_forecast_avoids_informed_flow():
    md, sig = informed_market()
    out = {}
    for name in ("join", "forecast"):
        q = ForecastQuoter(sig, 10 * MS, size=0.01, max_pos=1.0, tick=0.1, **VARIANTS[name])
        out[name] = summarise(Engine(md, q, 10 * MS, QUEUE).run())
    # join is repeatedly swept just before the price moves against it
    assert out["join"]["fills"] > 50 and out["join"]["adverse_1s_bps"] < -5
    # the forecast quoter pulls the exposed side in time and is far less adversely selected
    assert out["forecast"]["fills"] < out["join"]["fills"] / 2
    assert out["forecast"]["pnl_gross_usd"] > out["join"]["pnl_gross_usd"]
    # measured: join is swept on all 200 moves; the forecast quoter is never caught
    assert out["join"]["fills"] == 200 and out["forecast"]["fills"] == 0
