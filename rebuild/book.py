"""A price-level (L2) order book.

Quantities are absolute: setting a level to zero removes it. Original
strings are kept alongside the floats because OKX's checksum is computed
over the exchange's own string formatting, not over parsed numbers.
"""

from __future__ import annotations

from sortedcontainers import SortedDict

BID, ASK = 1, -1


class L2Book:
    __slots__ = ("bids", "asks")

    def __init__(self) -> None:
        # price -> (qty, price_str, qty_str)
        self.bids: SortedDict = SortedDict()
        self.asks: SortedDict = SortedDict()

    def clear(self) -> None:
        self.bids.clear()
        self.asks.clear()

    def side(self, side: int) -> SortedDict:
        return self.bids if side == BID else self.asks

    def set(self, side: int, px_str: str, qty_str: str) -> tuple[float, float]:
        """Apply one absolute level update. Returns (price, qty) as floats."""
        px, qty = float(px_str), float(qty_str)
        levels = self.bids if side == BID else self.asks
        if qty == 0.0:
            levels.pop(px, None)  # deleting an absent level is normal
        else:
            levels[px] = (qty, px_str, qty_str)
        return px, qty

    def load(self, bids, asks) -> None:
        """Replace the whole book from [price_str, qty_str, ...] lists."""
        self.clear()
        for lvl in bids:
            self.set(BID, lvl[0], lvl[1])
        for lvl in asks:
            self.set(ASK, lvl[0], lvl[1])

    # Reading -----------------------------------------------------------------
    def best_bid(self) -> float | None:
        return self.bids.peekitem(-1)[0] if self.bids else None

    def best_ask(self) -> float | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def crossed(self) -> bool:
        bb, ba = self.best_bid(), self.best_ask()
        return bb is not None and ba is not None and bb >= ba

    def top(self, n: int) -> tuple[list, list]:
        """([(px, qty), ...] best-first bids, same for asks)."""
        nb, na = min(n, len(self.bids)), min(n, len(self.asks))
        bids = [(p, v[0]) for p, v in (self.bids.peekitem(-1 - i) for i in range(nb))]
        asks = [(p, v[0]) for p, v in (self.asks.peekitem(i) for i in range(na))]
        return bids, asks

    def top_strings(self, n: int) -> tuple[list, list]:
        nb, na = min(n, len(self.bids)), min(n, len(self.asks))
        bids = [self.bids.peekitem(-1 - i)[1][1:] for i in range(nb)]
        asks = [self.asks.peekitem(i)[1][1:] for i in range(na)]
        return bids, asks

    def levels(self):
        """All levels as (side, px, qty), for writing a full reset."""
        for px, v in self.bids.items():
            yield BID, px, v[0]
        for px, v in self.asks.items():
            yield ASK, px, v[0]

    def copy(self) -> "L2Book":
        b = L2Book()
        b.bids = self.bids.copy()
        b.asks = self.asks.copy()
        return b
