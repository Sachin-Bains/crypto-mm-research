"""Tests for the market studies: event parsing conventions, episodes and signed price paths."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.events import BybitTickers, parse_binance, parse_bybit, parse_okx  # noqa: E402
from research.liquidations import GRID, S, episodes, signed_paths  # noqa: E402


def test_side_conventions_are_normalised_to_the_forced_order():
    liq, marks = [], []
    parse_binance(1, {"stream": "btcusdt@forceOrder", "data": {"e": "forceOrder", "E": 5, "o": {
        "s": "BTCUSDT", "S": "SELL", "q": "0.5", "z": "0.5", "p": "80000", "ap": "79990", "T": 4}}}, liq, marks)
    parse_bybit(2, {"topic": "allLiquidation.BTCUSDT", "ts": 9, "data": [
        {"T": 8, "s": "BTCUSDT", "S": "Buy", "v": "0.2", "p": "80000"}]}, liq, marks, BybitTickers())
    parse_okx(3, {"arg": {"channel": "liquidation-orders", "instType": "SWAP"}, "data": [
        {"instId": "BTC-USDT-SWAP", "details": [{"side": "buy", "posSide": "short", "sz": "30", "bkPx": "80100", "ts": "7"}]}]},
        liq, marks, {})
    assert [(r[0], r[4]) for r in liq] == [("binance", -1), ("bybit", -1), ("okx", 1)]
    assert liq[0][5] == 0.5 and liq[0][6] == 79990       # filled quantity, average price
    assert abs(liq[2][5] - 0.3) < 1e-12                   # 30 contracts of 0.01 BTC
    assert liq[0][3] == 4 * 1_000_000                     # exchange time in ns


def test_bybit_ticker_deltas_carry_fields_forward():
    marks, t = [], BybitTickers()
    t.update(1, {"ts": 1, "data": {"symbol": "BTCUSDT", "markPrice": "80000", "fundingRate": "0.0001",
                                   "nextFundingTime": "1000"}}, marks)
    t.update(2, {"ts": 2, "data": {"symbol": "BTCUSDT", "markPrice": "80010"}}, marks)
    assert marks[-1][4] == 80010 and marks[-1][6] == 0.0001


def test_episodes_group_within_two_seconds_and_take_the_net_direction():
    t = np.array([0, 1, 1.5, 10]) * S
    eps = episodes({"t": t.astype(np.int64), "side": np.array([-1, -1, 1, 1]), "usd": np.array([100, 100, 50, 10.0]),
                    "venue": np.array(["okx", "binance", "bybit", "bybit"])})
    assert len(eps) == 2 and eps[0]["dir"] == -1 and eps[0]["n"] == 3 and eps[0]["first"] == "okx"


def test_signed_path_is_positive_when_price_moves_with_the_forced_flow():
    ts = np.array([0, 5 * S, 20 * S], dtype=np.int64) + 100 * S
    mid = np.array([100.0, 99.9, 100.0])
    p = signed_paths(ts, mid, np.array([101 * S]), np.array([-1.0]), GRID)   # forced selling at 101 s
    i = int(np.argmin(np.abs(GRID - 5)))
    assert abs(p[0, i] - 10.0) < 1e-9                    # fell 10 bp: +10 in the forced direction


def test_xcorr_finds_a_known_lead():
    from research.leadlag import MAXLAG, llr, xcorr
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    b = np.r_[np.zeros(7), a[:-7]] + 0.5 * rng.normal(size=20000)   # b copies a 7 steps (70 ms) later
    c = xcorr(a, b)
    assert int(np.argmax(c)) - MAXLAG == 7 and llr(c) > 1
