"""Extract liquidations, mark price / funding, and clock samples from the raw recordings.

Usage:
  python -m research.events --dates 2026-10-15,2026-10-16

Writes data/clean/events/<date>/{liquidations,marks,clock}.parquet (date = UTC day of the
exchange event time where there is one, else of the receive time).

Conventions, normalised across venues:
  side  +1 = the forced order BUYS (a short was liquidated; pushes the price up)
        -1 = the forced order SELLS (a long was liquidated)
  Binance forceOrder: S is the forced order's side.
  Bybit allLiquidation: S is the side of the liquidated POSITION ("Buy" = a long was liquidated),
        so it is inverted.
  OKX liquidation-orders: side is the forced order's side; sz is in contracts (0.01 BTC, 0.1 ETH).
  Binance only pushes the latest liquidation per symbol per second, so its counts are a lower bound.
  The study checks these conventions against the aggressor side of each venue's own trades.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from recorder.storage import iter_file
from rebuild.sink import day_of

OKX_CTVAL = {"BTC-USDT-SWAP": 0.01, "ETH-USDT-SWAP": 0.1}
SYMBOL = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "BTC-USDT-SWAP": "BTC", "ETH-USDT-SWAP": "ETH"}
MS = 1_000_000

LIQ_SCHEMA = pa.schema([("venue", pa.string()), ("symbol", pa.string()), ("recv_ns", pa.int64()),
                        ("exch_ns", pa.int64()), ("side", pa.int8()), ("qty", pa.float64()), ("price", pa.float64())])
MARK_SCHEMA = pa.schema([("venue", pa.string()), ("symbol", pa.string()), ("recv_ns", pa.int64()), ("exch_ns", pa.int64()),
                         ("mark", pa.float64()), ("index", pa.float64()), ("funding_rate", pa.float64()),
                         ("next_funding_ns", pa.int64())])
CLOCK_SCHEMA = pa.schema([("venue", pa.string()), ("t0_ns", pa.int64()), ("t1_ns", pa.int64()), ("server_ns", pa.int64())])


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _i(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def parse_binance(recv: int, msg: dict, liq: list, marks: list) -> str | None:
    stream, d = msg.get("stream", ""), msg.get("data")
    if not isinstance(d, dict):
        return None
    if stream.endswith("@forceorder") or d.get("e") == "forceOrder":
        o = d.get("o", {})
        sym = SYMBOL.get(o.get("s"))
        if sym is None:
            return "liq_other_symbol"
        side = 1 if o.get("S") == "BUY" else -1 if o.get("S") == "SELL" else 0
        qty = _f(o.get("z")) if _f(o.get("z")) > 0 else _f(o.get("q"))
        price = _f(o.get("ap")) if _f(o.get("ap")) > 0 else _f(o.get("p"))
        liq.append(("binance", sym, recv, _i(o.get("T") or d.get("E")) * MS, side, qty, price))
        return "liq"
    if d.get("e") == "markPriceUpdate":
        sym = SYMBOL.get(d.get("s"))
        if sym:
            marks.append(("binance", sym, recv, _i(d.get("E")) * MS, _f(d.get("p")), _f(d.get("i")), _f(d.get("r")),
                          _i(d.get("T")) * MS))
            return "mark"
    return None


class BybitTickers:
    """Bybit sends a snapshot then deltas with only the changed fields; carry the rest forward."""

    def __init__(self):
        self.state = defaultdict(dict)

    def update(self, recv: int, msg: dict, marks: list) -> str | None:
        d = msg.get("data") or {}
        sym = SYMBOL.get(d.get("symbol"))
        if sym is None:
            return None
        st = self.state[sym]
        st.update({k: v for k, v in d.items() if v not in (None, "")})
        if "markPrice" not in st:
            return None
        marks.append(("bybit", sym, recv, _i(msg.get("ts")) * MS, _f(st.get("markPrice")), _f(st.get("indexPrice")),
                      _f(st.get("fundingRate")), _i(st.get("nextFundingTime")) * MS))
        return "mark"


def parse_bybit(recv: int, msg: dict, liq: list, marks: list, tick: BybitTickers) -> str | None:
    topic = msg.get("topic", "")
    if topic.startswith("allLiquidation."):
        n = 0
        for x in msg.get("data") or []:
            sym = SYMBOL.get(x.get("s"))
            if sym is None:
                continue
            side = -1 if x.get("S") == "Buy" else 1 if x.get("S") == "Sell" else 0   # position side, inverted
            liq.append(("bybit", sym, recv, _i(x.get("T")) * MS, side, _f(x.get("v")), _f(x.get("p"))))
            n += 1
        return "liq" if n else "liq_other_symbol"
    if topic.startswith("tickers."):
        return tick.update(recv, msg, marks)
    return None


def parse_okx(recv: int, msg: dict, liq: list, marks: list, okx_state: dict) -> str | None:
    arg = msg.get("arg") or {}
    ch = arg.get("channel")
    data = msg.get("data")
    if not isinstance(data, list):
        return None
    if ch == "liquidation-orders":
        n = 0
        for item in data:
            inst = item.get("instId")
            sym = SYMBOL.get(inst)
            if sym is None:
                continue
            for x in item.get("details") or []:
                side = 1 if x.get("side") == "buy" else -1 if x.get("side") == "sell" else 0
                liq.append(("okx", sym, recv, _i(x.get("ts")) * MS, side, _f(x.get("sz")) * OKX_CTVAL[inst], _f(x.get("bkPx"))))
                n += 1
        return "liq" if n else "liq_other_symbol"
    if ch in ("mark-price", "funding-rate"):
        for x in data:
            sym = SYMBOL.get(x.get("instId"))
            if sym is None:
                continue
            st = okx_state.setdefault(sym, {})
            if ch == "mark-price":
                st["mark"], st["ts"] = _f(x.get("markPx")), _i(x.get("ts"))
            else:
                st["fr"], st["next"] = _f(x.get("fundingRate")), _i(x.get("fundingTime"))
            if "mark" in st:
                marks.append(("okx", sym, recv, st.get("ts", 0) * MS, st["mark"], float("nan"), st.get("fr", float("nan")),
                              st.get("next", 0) * MS))
        return "mark"
    return None


def clock_row(venue: str, payload: str):
    c = orjson.loads(payload)
    body = c.get("body") or {}
    if venue == "binance":
        server = _i(body.get("serverTime")) * MS
    elif venue == "bybit":
        server = _i((body.get("result") or {}).get("timeNano")) or _i(body.get("time")) * MS
    else:
        d = body.get("data") or [{}]
        server = _i(d[0].get("ts")) * MS
    return (venue, int(c["t0"]), int(c["t1"]), server)


def extract(raw: Path, days: list[str]):
    from datetime import date, timedelta
    want = set(days)
    # a day's events can sit in the last hour of the previous receive-date folder or the first
    # hour of the next one, so those two files are read as well
    files = set()
    for d in days:
        y = date.fromisoformat(d)
        files |= {(d, "*"), ((y - timedelta(days=1)).isoformat(), "23"), ((y + timedelta(days=1)).isoformat(), "00")}
    if any(f[1] == "*" for f in files):
        files = {f for f in files if f[1] == "*" or (f[0], "*") not in files}
    out = {d: {"liq": [], "mark": [], "clock": []} for d in days}
    counts = Counter()
    for venue in ("binance", "bybit", "okx"):
        tick, okx_state = BybitTickers(), {}
        for folder, hour in sorted(files):
            for path in sorted((raw / venue / folder).glob(f"{hour}.ndjson.zst")):
                for recv, tag, payload in iter_file(path):
                    if tag == "clock":
                        try:
                            row = clock_row(venue, payload)
                        except Exception:
                            counts[f"{venue}:clock_unparsed"] += 1
                            continue
                        dd = day_of(row[1])
                        if dd in want:
                            out[dd]["clock"].append(row)
                        continue
                    if not ('iquidat' in payload or 'markPrice' in payload or 'tickers.' in payload
                            or 'mark-price' in payload or 'funding-rate' in payload or 'forceOrder' in payload):
                        continue
                    try:
                        msg = orjson.loads(payload)
                    except orjson.JSONDecodeError:
                        counts[f"{venue}:bad_json"] += 1
                        continue
                    if not isinstance(msg, dict):
                        continue
                    liq, marks = [], []
                    if venue == "binance":
                        kind = parse_binance(recv, msg, liq, marks)
                    elif venue == "bybit":
                        kind = parse_bybit(recv, msg, liq, marks, tick)
                    else:
                        kind = parse_okx(recv, msg, liq, marks, okx_state)
                    if kind:
                        counts[f"{venue}:{kind}"] += 1
                    for r in liq:
                        dd = day_of(r[3] if r[3] > 0 else r[2])
                        if dd in want:
                            out[dd]["liq"].append(r)
                    for r in marks:
                        dd = day_of(r[2])
                        if dd in want:
                            out[dd]["mark"].append(r)
    return out, counts


def write(out_root: Path, out: dict) -> None:
    for d, tabs in out.items():
        base = out_root / d
        base.mkdir(parents=True, exist_ok=True)
        for name, rows, schema in (("liquidations", tabs["liq"], LIQ_SCHEMA), ("marks", tabs["mark"], MARK_SCHEMA),
                                   ("clock", tabs["clock"], CLOCK_SCHEMA)):
            cols = list(zip(*rows)) if rows else [[] for _ in schema]
            pq.write_table(pa.table({f.name: pa.array(c, type=f.type) for f, c in zip(schema, cols)}),
                           base / f"{name}.parquet", compression="zstd")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", required=True)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/clean/events")
    args = ap.parse_args()
    days = [d.strip() for d in args.dates.split(",") if d.strip()]
    out, counts = extract(Path(args.raw), days)
    write(Path(args.out), out)
    print("messages parsed:", ", ".join(f"{k} {v:,}" for k, v in sorted(counts.items())))
    for d in days:
        liq = out[d]["liq"]
        by = Counter((r[0], r[1]) for r in liq)
        usd = defaultdict(float)
        for r in liq:
            usd[(r[0], r[1])] += r[5] * r[6]
        print(f"{d}: liquidations " + ", ".join(f"{v} {s} {by[(v, s)]:,} (${usd[(v, s)] / 1e6:.1f}m)" for v, s in sorted(by))
              + f"; mark rows {len(out[d]['mark']):,}; clock samples {len(out[d]['clock']):,}")
    print(f"Saved to {args.out}/<date>/")


if __name__ == "__main__":
    main()
