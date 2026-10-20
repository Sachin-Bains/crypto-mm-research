"""Forecasts and intervals made available to a strategy inside the simulator, without look-ahead.

The simulator runs on the exchange clock; forecasts were built on our receive
clock. A strategy at simulator time t has seen Binance up to exchange time
t - latency. We find the moment our machine received that Binance update and
use the latest forecast made at or before that receive time. So the strategy
only ever uses a forecast built from information it could actually have had.

Limitation, stated plainly: forecasts were built from what reached Glasgow, so
in the 10 ms and 1 ms scenarios only order timing improves; the forecast's
information set is still the Glasgow one.

Besides the forecast f (bps) and the interval half-width w (bps), the signal
reports rho = w / typical w, where typical w is a causal exponentially weighted
average with a 10-minute half-life. rho > 1 means more uncertain than usual.
"""

from __future__ import annotations

import math

import numpy as np


class Signal:
    def __init__(self, sig_recv: np.ndarray, pred: np.ndarray, hw: np.ndarray,
                 bbo_exch: np.ndarray, bbo_recv: np.ndarray,
                 halflife_s: float = 600.0, max_age_ns: int = 1_000_000_000,
                 vol: np.ndarray | None = None) -> None:
        order = np.argsort(sig_recv, kind="stable")
        self.t = np.asarray(sig_recv, dtype=np.int64)[order]
        self.f = np.asarray(pred, dtype=np.float64)[order]
        self.w = np.asarray(hw, dtype=np.float64)[order]
        o = np.argsort(bbo_exch, kind="stable")
        self.bx = np.asarray(bbo_exch, dtype=np.int64)[o]
        # receive time of the latest Binance update seen up to each exchange time
        self.br = np.maximum.accumulate(np.asarray(bbo_recv, dtype=np.int64)[o]) if len(o) else self.bx
        self.max_age = max_age_ns
        self.typ = self._ewma(self.w, halflife_s)
        # optional realised volatility on the same rows, for the volatility-filter control
        self.v = np.asarray(vol, dtype=np.float64)[order] if vol is not None else None
        self.vtyp = self._ewma(self.v, halflife_s) if vol is not None else None

    def _ewma(self, x: np.ndarray, halflife_s: float) -> np.ndarray:
        out = np.full(len(self.t), np.nan)
        m, last_t = math.nan, None
        k = math.log(2) / (halflife_s * 1e9)
        for i, (ti, wi) in enumerate(zip(self.t.tolist(), x.tolist())):
            if math.isfinite(wi) and wi > 0:
                if math.isnan(m):
                    m = wi
                else:
                    a = 1.0 - math.exp(-k * (ti - last_t))
                    m = m + a * (wi - m)
                last_t = ti
            out[i] = m
        return out

    def window(self) -> tuple[int, int] | None:
        """Receive-time span where forecasts and intervals exist."""
        ok = np.isfinite(self.f) & np.isfinite(self.w)
        if not ok.any():
            return None
        return int(self.t[ok][0]), int(self.t[ok][-1])

    def _row(self, seen_exch_ns: int) -> int | None:
        """Index of the latest row known to a strategy that has seen Binance up to this exchange
        time, or None if there is none or it is stale."""
        i = int(np.searchsorted(self.bx, seen_exch_ns, side="right")) - 1
        if i < 0:
            return None
        r = int(self.br[i])
        j = int(np.searchsorted(self.t, r, side="right")) - 1
        if j < 0 or r - self.t[j] > self.max_age:
            return None
        return j

    def vol_ratio(self, seen_exch_ns: int):
        """Realised volatility over its causal typical level, or None (no volatility series,
        no fresh row)."""
        if self.v is None:
            return None
        j = self._row(seen_exch_ns)
        if j is None:
            return None
        v, typ = self.v[j], self.vtyp[j]
        if not (math.isfinite(v) and math.isfinite(typ)) or typ <= 0:
            return None
        return float(v / typ)

    def at(self, seen_exch_ns: int):
        """(forecast_bps, half_width_bps, rho) known to a strategy that has seen Binance up to
        this exchange time, or None if no fresh, finite forecast exists."""
        j = self._row(seen_exch_ns)
        if j is None:
            return None
        f, w, typ = self.f[j], self.w[j], self.typ[j]
        if not (math.isfinite(f) and math.isfinite(w) and math.isfinite(typ)) or typ <= 0:
            return None
        return float(f), float(w), float(w / typ)
