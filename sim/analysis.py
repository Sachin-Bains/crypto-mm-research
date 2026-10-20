"""Turn a simulation result into numbers: fills, markouts and PnL attribution.

PnL (before fees) splits exactly into two parts:
  spread captured  sum over fills of side * (mid at fill - fill price) * qty
                   what you earn for providing liquidity at the moment you trade
  inventory        everything else: how the mid moved while you held a position.
                   For a market maker this is mostly adverse selection.

For each fill and horizon h (1, 5 and 30 seconds), two numbers, both in basis
points of traded notional and signed so that negative means it went against you:
  markout   side * (mid(t + h) - fill price): what the fill was worth h seconds later
  adverse   side * (mid(t + h) - mid(t)): the price move alone, i.e. adverse selection
markout = spread captured + adverse.
"""

from __future__ import annotations

import math

import numpy as np

from .engine import Result

HORIZONS_S = (1, 5, 30)

# Fee tiers in basis points of notional for maker (passive) fills.
# Retail rates are the published base tiers on Binance, Bybit and OKX (2026).
# The rebate tier is illustrative of exchange market-maker programmes, not a quoted rate.
FEE_TIERS_BPS = {"retail (2.0 bp)": 2.0, "top VIP (0 bp)": 0.0, "MM rebate (-0.5 bp)": -0.5}


def mid_at(res: Result, t_ns: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(res.mid_t, t_ns, side="right") - 1
    out = np.full(len(t_ns), np.nan)
    ok = (idx >= 0) & (t_ns <= res.mid_t[-1]) if len(res.mid_t) else np.zeros(len(t_ns), bool)
    out[ok] = res.mid_v[idx[ok]]
    return out


def summarise(res: Result) -> dict:
    f = res.fills
    out = {
        "model": res.model,
        "latency_ms": res.latency_ns / 1e6,
        "orders_placed": res.orders_placed,
        "orders_rejected": res.orders_rejected,
        "fills": len(f),
        "volume": round(sum(x.qty for x in f), 6),
        "notional_usd": round(res.notional, 2),
    }
    if not f or res.mid_v is None or not len(res.mid_v):
        # no trades: nothing earned or lost
        out.update(pnl_gross_usd=0.0, pnl_gross_bps=0.0, spread_captured_usd=0.0, inventory_usd=0.0)
        for name in FEE_TIERS_BPS:
            out[f"pnl_net_usd [{name}]"] = 0.0
        return out
    t = np.array([x.t_ns for x in f], dtype=np.int64)
    side = np.array([x.side for x in f], dtype=np.float64)
    px = np.array([x.price for x in f])
    q = np.array([x.qty for x in f])
    m0 = np.array([x.mid for x in f])

    final_mid = res.mid_v[-1]
    gross = res.cash + res.position * final_mid
    spread = float(np.nansum(side * (m0 - px) * q))
    notional = max(res.notional, 1e-12)
    out.update({
        "final_position": round(res.position, 6),
        "pnl_gross_usd": round(gross, 4),
        "spread_captured_usd": round(spread, 4),
        "inventory_usd": round(gross - spread, 4),
        "pnl_gross_bps": round(gross / notional * 1e4, 3),
        "spread_captured_bps": round(spread / notional * 1e4, 3),
        "median_wait_s": round(float(np.median([x.wait_ns for x in f])) / 1e9, 3),
        "fill_share_of_orders": round(len({x.oid for x in f}) / max(res.orders_placed, 1), 4),
    })
    for h in HORIZONS_S:
        mh = mid_at(res, t + int(h * 1e9))
        ok = ~np.isnan(mh)
        if not ok.any():
            out[f"markout_{h}s_bps"] = out[f"adverse_{h}s_bps"] = math.nan
            continue
        w = float(np.sum(q[ok] * px[ok]))
        out[f"markout_{h}s_bps"] = round(float(np.sum(side[ok] * (mh[ok] - px[ok]) * q[ok])) / w * 1e4, 3)
        out[f"adverse_{h}s_bps"] = round(float(np.sum(side[ok] * (mh[ok] - m0[ok]) * q[ok])) / w * 1e4, 3)
    for name, bps in FEE_TIERS_BPS.items():
        out[f"pnl_net_usd [{name}]"] = round(gross - bps * 1e-4 * res.notional, 4)
    return out


def hourly_markout_usd(res: Result, h_s: float = 5.0) -> dict[int, float]:
    """5 s markout in USD summed per clock hour of fill time: side * (mid(t + h) - price) * qty.

    Used to compare two strategies hour by hour on the same data (a paired test with hours as
    blocks), which accounts for fills within an hour being correlated with each other.
    """
    f = res.fills
    if not f or res.mid_v is None or not len(res.mid_v):
        return {}
    t = np.array([x.t_ns for x in f], dtype=np.int64)
    side = np.array([x.side for x in f], dtype=np.float64)
    px = np.array([x.price for x in f])
    q = np.array([x.qty for x in f])
    mh = mid_at(res, t + int(h_s * 1e9))
    ok = ~np.isnan(mh)
    hour = t[ok] // (3600 * 10**9)
    v = side[ok] * (mh[ok] - px[ok]) * q[ok]
    return {int(k): float(v[hour == k].sum()) for k in np.unique(hour)}


def hourly_notional_usd(res: Result, h_s: float = 5.0) -> dict[int, float]:
    """Traded notional per clock hour, over the same fills hourly_markout_usd counts."""
    f = res.fills
    if not f or res.mid_v is None or not len(res.mid_v):
        return {}
    t = np.array([x.t_ns for x in f], dtype=np.int64)
    n = np.array([x.price * x.qty for x in f])
    ok = ~np.isnan(mid_at(res, t + int(h_s * 1e9)))
    hour = t[ok] // (3600 * 10**9)
    return {int(k): float(n[ok][hour == k].sum()) for k in np.unique(hour)}


def paired_per_fill(base_m: dict[int, float], base_n: dict[int, float],
                    other_m: dict[int, float], other_n: dict[int, float]) -> tuple[float, float, int]:
    """Per-fill quality test: in each hour where both traded, 5 s markout in bp of notional,
    other minus base; returns the mean difference (bp), its t statistic and the number of hours.
    Unlike the dollar comparison, trading less does not improve this number."""
    hours = sorted(k for k in set(base_n) & set(other_n) if base_n[k] > 0 and other_n[k] > 0)
    if len(hours) < 2:
        return math.nan, math.nan, len(hours)
    d = np.array([1e4 * (other_m.get(k, 0.0) / other_n[k] - base_m.get(k, 0.0) / base_n[k]) for k in hours])
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else math.nan
    return float(d.mean()), t, len(d)


def paired_vs(base: dict[int, float], other: dict[int, float]) -> tuple[float, float, int]:
    """Mean hourly difference (other - base) in USD, its t statistic, and the number of hours.
    An hour with no fills for a strategy counts as 0 USD for it."""
    hours = sorted(set(base) | set(other))
    if len(hours) < 2:
        return math.nan, math.nan, len(hours)
    d = np.array([other.get(k, 0.0) - base.get(k, 0.0) for k in hours])
    sd = float(d.std(ddof=1))
    t = float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else math.nan
    return float(d.mean()), t, len(d)
