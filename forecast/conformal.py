"""Prediction intervals for the forecasts, and how well they hold up when the market shifts.

Every method gives a symmetric interval  prediction +/- half_width  aiming to
contain the outcome a share (1 - alpha) of the time. Methods:

  static      split conformal calibrated once on the first block of out-of-sample
              residuals, then frozen. The "fit once, deploy" baseline.
  rolling     split conformal recalibrated every `update` on the residuals of the
              trailing `window`.
  scaled      as rolling, but on residuals divided by recent volatility (rv_30s), so
              intervals widen and narrow with the market. Locally adaptive conformal.
  regime      scaled, calibrated separately within volatility terciles of the window
              (calm, normal, stressed). Mondrian conformal.
  adaptive    adaptive conformal inference (Gibbs and Candes, 2021) on scaled residuals:
              the working miscoverage level alpha_t moves after every update,
              alpha_t+1 = alpha_t + gamma * (alpha - miss rate since last update),
              so persistent under-coverage widens intervals and over-coverage narrows them.

Label delay. A forecast made at t for horizon h is only scored at t + h. Every
calibration set at time u contains only rows with t + h <= u, and the adaptive
update only counts misses whose outcomes have arrived. Without this, intervals
would be calibrated on outcomes not yet known.

Quantiles use the finite-sample conformal rank ceil((n + 1)(1 - alpha)).
"""

from __future__ import annotations

import math

import numpy as np

METHODS = ("static", "rolling", "scaled", "regime", "adaptive")


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Smallest score s such that at least ceil((n+1)(1-alpha)) of n scores are <= s."""
    n = len(scores)
    if n == 0 or alpha <= 0:
        return math.inf
    if alpha >= 1:
        return 0.0
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return math.inf
    return float(np.partition(scores, k - 1)[k - 1])


def intervals(t: np.ndarray, y: np.ndarray, pred: np.ndarray, scale: np.ndarray, h_ns: int,
              alpha: float = 0.1, window_ns: int = 600 * 10**9, update_ns: int = 10**9,
              static_ns: int = 3600 * 10**9, gamma: float = 0.005, min_cal: int = 300) -> dict[str, np.ndarray]:
    """Half-widths per row for every method. Rows without enough calibration history get NaN.

    t must be sorted. y may contain NaN (no outcome); those rows never enter calibration.
    """
    n = len(t)
    r = np.abs(y - pred)
    ok = np.isfinite(r) & np.isfinite(scale) & (scale > 0)
    s = np.where(ok, r / np.where(ok, scale, 1.0), np.nan)
    out = {m: np.full(n, np.nan) for m in METHODS}

    # static: first `static_ns` of rows with outcomes, frozen afterwards
    t_first = t[np.argmax(ok)] if ok.any() else t[0]
    cal = ok & (t < t_first + static_ns)
    q_static = conformal_quantile(r[cal], alpha) if cal.sum() >= min_cal else math.inf
    after = t >= t_first + static_ns + h_ns   # no overlap with its own calibration outcomes
    out["static"][after] = q_static

    step = (t - t[0]) // update_ns
    starts = np.flatnonzero(np.r_[True, step[1:] != step[:-1]])
    ends = np.r_[starts[1:], n]
    alpha_t = alpha
    prev_u = None
    last = {m: math.nan for m in ("rolling", "scaled", "adaptive")}
    last_regime = None

    for a, b in zip(starts, ends):
        u = t[0] + step[a] * update_ns      # interval for rows in [u, u + update) set at time u

        # adaptive update: misses among rows whose outcome arrived in (prev_u, u]
        if prev_u is not None:
            lo = np.searchsorted(t, prev_u - h_ns, side="right")
            hi = np.searchsorted(t, u - h_ns, side="right")
            hw = out["adaptive"][lo:hi]
            done = ok[lo:hi] & np.isfinite(hw)
            if done.any():
                miss = float(np.mean(r[lo:hi][done] > hw[done]))
                alpha_t = alpha_t + gamma * (alpha - miss)
        prev_u = u

        # calibration set: outcomes known by u, within the trailing window
        lo = np.searchsorted(t, u - h_ns - window_ns, side="left")
        hi = np.searchsorted(t, u - h_ns, side="right")
        cal = ok[lo:hi]
        if cal.sum() >= min_cal:
            rc, sc, vc = r[lo:hi][cal], s[lo:hi][cal], scale[lo:hi][cal]
            last["rolling"] = conformal_quantile(rc, alpha)
            last["scaled"] = conformal_quantile(sc, alpha)
            last["adaptive"] = conformal_quantile(sc, alpha_t)
            c1, c2 = np.quantile(vc, [1 / 3, 2 / 3])
            bins = np.digitize(vc, [c1, c2])
            qs = [conformal_quantile(sc[bins == k], alpha) if (bins == k).sum() >= min_cal // 3 else last["scaled"]
                  for k in range(3)]
            last_regime = (c1, c2, qs)

        sl = slice(a, b)
        out["rolling"][sl] = last["rolling"]
        out["scaled"][sl] = last["scaled"] * scale[sl]
        out["adaptive"][sl] = last["adaptive"] * scale[sl]
        if last_regime is not None:
            c1, c2, qs = last_regime
            k = np.digitize(scale[sl], [c1, c2])
            out["regime"][sl] = np.asarray(qs)[k] * scale[sl]
    return out


def evaluate(t: np.ndarray, y: np.ndarray, pred: np.ndarray, hw: np.ndarray, vol: np.ndarray,
             alpha: float = 0.1, bucket_ns: int = 600 * 10**9) -> dict:
    """Coverage, width and interval score, overall, by volatility regime and over time."""
    ok = np.isfinite(y) & np.isfinite(pred) & np.isfinite(hw) & np.isfinite(vol)
    if ok.sum() < 100:
        return {"n": int(ok.sum())}
    r, w, v, tt = np.abs(y - pred)[ok], hw[ok], vol[ok], t[ok]
    hit = r <= w
    out = {
        "n": int(ok.sum()),
        "coverage": round(float(hit.mean()), 4),
        "width_bps": round(float(np.mean(2 * w[np.isfinite(w)])), 4) if np.isfinite(w).any() else math.inf,
        "interval_score": round(float(np.mean(2 * w + (2 / alpha) * np.maximum(0.0, r - w))), 4),
    }
    q1, q2, q95 = np.quantile(v, [1 / 3, 2 / 3, 0.95])
    for name, m in (("calm", v <= q1), ("normal", (v > q1) & (v <= q2)), ("stressed", v > q2), ("top5pct_vol", v > q95)):
        out[f"cov_{name}"] = round(float(hit[m].mean()), 4) if m.any() else math.nan
    buckets = (tt - tt[0]) // bucket_ns
    cov_b = np.array([hit[buckets == k].mean() for k in np.unique(buckets) if (buckets == k).sum() >= 600])
    if len(cov_b):
        out["worst_10min_cov"] = round(float(cov_b.min()), 4)
        out["share_10min_below_80"] = round(float(np.mean(cov_b < 0.8)), 4)
    return out
