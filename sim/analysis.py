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
