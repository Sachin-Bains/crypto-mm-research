"""Tests for the Avellaneda-Stoikov benchmark and its calibration."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.data import MarketData  # noqa: E402
from sim.engine import ASK, BID, Api  # noqa: E402
from sim.strategies import AvellanedaStoikovQuoter  # noqa: E402

S = 1_000_000_000


class Always:
    def at(self, t):
        return (0.0, 1.0, 1.0)


def warm(q: AvellanedaStoikovQuoter, bp_per_s: float, mid=80_000.0, n=200):
    """Feed a mid that alternates up and down by bp_per_s each second: variance rate bp_per_s^2."""
    m = mid
    for i in range(n):
        m = mid * (1 + (bp_per_s if i % 2 else 0.0) / 1e4)
        q._update_vol(i * S, m)
    return m


def make(**kw):
    q = AvellanedaStoikovQuoter(Always(), 0, tick=0.1, max_pos=0.05, size=0.01, **kw)
    q.bind(Api())
    return q


def test_variance_rate_is_estimated():
    q = make()
    warm(q, 2.0)
    assert abs(q.sigma2() - 4.0) < 0.2


def test_flat_inventory_low_gamma_quotes_at_the_touch():
    q = make(gamma=1e-4, k=1e4)          # half-spread ~ 1/k = 0.0001 bp, well inside one tick
    warm(q, 1.0)
    want = q.desired(200 * S, 80_000.0, 80_000.1)
    assert want == {BID: 80_000.0, ASK: 80_000.1}


def test_long_inventory_leans_the_bid_back_and_keeps_the_ask_passive():
    q = make(gamma=0.01, k=1e4, spread=False, tau_s=30)
    warm(q, 1.0)                         # sigma^2 ~ 1 bp^2/s
    q.pos = 0.03                         # 3 units long: lean = 3 * 0.01 * 1 * 30 = 0.9 bp, about 72 ticks at 80k
    want = q.desired(200 * S, 80_000.0, 80_000.1)
    lean_ticks = 0.9 / 1e4 * 80_000 / 0.1
    assert want[ASK] == 80_000.1         # can't sell below the best ask without crossing
    assert 80_000.0 - want[BID] >= (math.floor(lean_ticks) - 1) * 0.1


def test_textbook_spread_pushes_quotes_behind_the_touch():
    q = make(gamma=0.001, k=1.0)         # 1/k = 1 bp half-spread, about 80 ticks at 80k
    warm(q, 1.0)
    want = q.desired(200 * S, 80_000.0, 80_000.1)
    assert want[BID] <= 80_000.0 - 0.7 and want[ASK] >= 80_000.1 + 0.7


def test_no_quotes_before_volatility_is_known():
    q = make()
    assert q.desired(0, 80_000.0, 80_000.1) == {BID: None, ASK: None}


def test_calibration_recovers_known_decay():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from calibrate_as import reach
    rng = np.random.default_rng(0)
    n = 20_000
    mid = 80_000.0
    k_true = 20.0                         # per bp
    dist = rng.exponential(1 / k_true, size=2 * n)   # bp beyond the mid each second, per side
    sec = np.repeat(np.arange(n), 2)
    side = np.tile([1, -1], n)
    price = mid * (1 + side * dist / 1e4)
    t = sec * S + 500_000_000
    md = MarketData.from_arrays(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                                t, price, np.ones(2 * n), side,
                                bbo=(np.arange(n) * S, np.full(n, mid), np.ones(n), np.full(n, mid), np.ones(n)))
    r, m = reach(md)
    grid = np.linspace(0.01, 0.3, 30)
    lam = np.array([(r >= g).mean() for g in grid])
    slope = np.polyfit(grid, np.log(lam), 1)[0]
    assert abs(-slope - k_true) < 1.5
