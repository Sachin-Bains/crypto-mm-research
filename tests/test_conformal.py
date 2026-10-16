"""Tests for the interval methods.

Coverage targets are 90% throughout. Synthetic residuals are drawn from known
distributions so the right answer is known in advance.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forecast.conformal import METHODS, conformal_quantile, evaluate, intervals  # noqa: E402

S = 10**9
STEP = 10**8   # 100 ms rows


def series(hours, sd_fn, scale_fn, seed=0, h=5):
    rng = np.random.default_rng(seed)
    n = int(hours * 36000)
    t = np.arange(n, dtype=np.int64) * STEP
    sd, scale = sd_fn(t), scale_fn(t)
    pred = np.zeros(n)
    y = rng.normal(0, 1, n) * sd
    return t, y, pred, scale, h * S


def cov(y, pred, hw, mask):
    m = mask & np.isfinite(hw)
    return float(np.mean(np.abs(y - pred)[m] <= hw[m]))


def test_conformal_quantile_rank():
    s = np.arange(1, 10, dtype=float)          # n = 9: rank ceil(10 * 0.9) = 9
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s, 0.2) == 8.0   # ceil(10 * 0.8) = 8
    assert conformal_quantile(np.arange(5.0), 0.1) == math.inf   # too few points for 90%


def test_all_methods_cover_when_nothing_changes():
    t, y, pred, scale, h = series(3, lambda t: np.ones(len(t)), lambda t: np.ones(len(t)))
    hw = intervals(t, y, pred, scale, h)
    late = t > 3600 * S + h
    for m in METHODS:
        c = cov(y, pred, hw[m], late)
        assert abs(c - 0.9) < 0.02, (m, c)


def test_volatility_shift_breaks_static_not_scaled():
    """Noise quadruples after 2 hours and recent volatility (the scale) shows it."""
    jump = lambda t: np.where(t >= 2 * 3600 * S, 4.0, 1.0)   # noqa: E731
    t, y, pred, scale, h = series(3, jump, jump, seed=1)
    hw = intervals(t, y, pred, scale, h)
    after = t >= 2 * 3600 * S
    assert cov(y, pred, hw["static"], after) < 0.5
    for m in ("scaled", "regime", "adaptive"):
        assert abs(cov(y, pred, hw[m], after) - 0.9) < 0.03, m
    # the rolling window recovers, but only after it fills with post-shift residuals
    first10 = after & (t < 2 * 3600 * S + 600 * S)
    assert cov(y, pred, hw["rolling"], first10) < 0.8


def test_adaptive_recovers_from_shift_the_scale_misses():
    """Noise triples after 2 hours but volatility does NOT show it (an unobserved shift).
    Scaled conformal is miscalibrated until its window refills; adaptive corrects faster."""
    t, y, pred, scale, h = series(3, lambda t: np.where(t >= 2 * 3600 * S, 3.0, 1.0),
                                  lambda t: np.ones(len(t)), seed=2)
    hw = intervals(t, y, pred, scale, h)
    # measured (5 s horizon): 1 to 3 minutes after the shift, scaled ~56%, adaptive ~84%; static ~41%
    shock = (t >= 2 * 3600 * S + 60 * S) & (t < 2 * 3600 * S + 180 * S)
    assert cov(y, pred, hw["scaled"], shock) < 0.7
    assert cov(y, pred, hw["adaptive"], shock) > 0.8
    # once the window has refilled, both are back on target; static never recovers
    later = t >= 2 * 3600 * S + 900 * S
    assert abs(cov(y, pred, hw["scaled"], later) - 0.9) < 0.03
    assert cov(y, pred, hw["static"], later) < 0.5


def test_no_look_ahead_through_label_delay():
    """Changing outcomes after time T must not change any interval set at or before T."""
    t, y, pred, scale, h = series(2, lambda t: np.ones(len(t)), lambda t: np.ones(len(t)), seed=3, h=30)
    T = int(1.5 * 3600 * S)
    y2 = y.copy()
    y2[t > T - h] *= 10.0      # outcomes arriving after T (row t's outcome arrives at t + h)
    a = intervals(t, y, pred, scale, h)
    b = intervals(t, y2, pred, scale, h)
    before = t <= T
    for m in METHODS:
        np.testing.assert_array_equal(a[m][before], b[m][before], err_msg=m)
    assert not np.allclose(np.nan_to_num(a["rolling"][~before]), np.nan_to_num(b["rolling"][~before]))


def test_evaluate_reports_regimes_and_windows():
    t, y, pred, scale, h = series(1, lambda t: np.ones(len(t)), lambda t: 1 + (t % (600 * S)) / (600 * S))
    hw = np.full(len(t), 1.645)
    out = evaluate(t, y, pred, hw, scale)
    assert abs(out["coverage"] - 0.9) < 0.01
    assert out["width_bps"] == round(2 * 1.645, 4)
    assert {"cov_calm", "cov_stressed", "worst_10min_cov", "share_10min_below_80"} <= out.keys()


def test_adaptive_never_stalls_after_a_burst_of_misses():
    """A long burst of large errors drives the working level to its floor. The interval must stay
    finite (capped at the widest calibration residual) and the method must recover afterwards.
    Before the fix, infinite intervals were left out of the update, so it stayed infinite for good."""
    burst = lambda t: np.where((t >= 2 * 3600 * S) & (t < 2 * 3600 * S + 600 * S), 20.0, 1.0)   # noqa: E731
    t, y, pred, scale, h = series(4, burst, lambda t: np.ones(len(t)), seed=4)
    hw = intervals(t, y, pred, scale, h)
    live = t >= 3600 * S
    assert np.isfinite(hw["adaptive"][live]).all()
    later = t >= 3 * 3600 * S
    assert abs(cov(y, pred, hw["adaptive"], later) - 0.9) < 0.03


def test_evaluate_counts_infinite_intervals_as_covering():
    t, y, pred, scale, h = series(1, lambda t: np.ones(len(t)), lambda t: np.ones(len(t)))
    hw = np.full(len(t), 1.645)
    hw[: len(t) // 2] = np.inf
    out = evaluate(t, y, pred, hw, scale)
    assert out["share_infinite"] == 0.5 and out["width_bps"] == math.inf
    assert out["coverage"] > 0.94       # half always covered, half about 90%
