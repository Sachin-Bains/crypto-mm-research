# crypto-mm-research

Research stack for one question: **when the market's distribution shifts, does a model's uncertainty stay honest, and can a market maker use honest uncertainty to lose less to informed flow?**

This repository grows layer by layer:

1. **Recorder**: raw order book, trade, mark price, funding and liquidation data for BTC and ETH perpetuals on Binance, Bybit and OKX.
2. **Simulator**, in progress:
   - **Book rebuild** (done): validated order books and normalised trades from the raw data.
   - Fill model with queue position, latency and fees.
3. Short-horizon forecasts with conformal prediction intervals, tested under distribution shift.
4. Uncertainty-aware market maker, evaluated with PnL attribution and markouts.
5. Interactive results explorer and research note.

---

## Layer 1: the recorder

### What it records

| Venue | Order book | Also |
|---|---|---|
| Binance USD-M | diff depth @100ms + bookTicker, REST snapshots | aggTrade, markPrice@1s (mark, index, funding), forceOrder |
| Bybit linear | orderbook.200 (snapshot + deltas, 100ms) | publicTrade, tickers, allLiquidation |
| OKX swap | books (400 levels, snapshot + updates, 100ms) | trades, funding-rate, mark-price, liquidation-orders |

Plus, per venue every 60 seconds, a REST server-time sample (for clock offset and feed latency) and instrument metadata every 6 hours (tick size, lot size, OKX contract value).

### Design decisions

- **Store raw, rebuild offline.** Messages are stored exactly as received, with the local receive time in nanoseconds. Order books are reconstructed later from the raw data. A bug in reconstruction can be fixed and re-run; a bug in a recorder that only stored processed books would lose data permanently.
- **Crash-tolerant files.** Each flush (every 2 seconds) appends an independent zstd frame. A power cut loses at most 2 seconds and never corrupts earlier data. Parquet is not used for capture because an unclosed Parquet file is unreadable; conversion happens offline.
- **Write before parse.** Every message is written to disk before any parsing, so a parsing error can never cost data.
- **Sequencing checks follow each venue's documented fields.**
  - Binance: each depth event's `pu` must equal the previous `u`. On a gap, a fresh REST snapshot is fetched.
  - OKX: `prevSeqId` must equal the previous `seqId`. On a gap, the book is resubscribed for a new snapshot (rate-limited to stay inside OKX's subscribe limits).
  - Bybit does not document `u` as strictly incrementing, so jumps are measured, not acted on. The first live run settles the question.
- **Regular resync points.** Binance REST snapshots every 30 minutes; Bybit and OKX books resubscribed every 60 minutes. The offline rebuild never has to trust a book for more than an hour without a checkpoint.
- **Binance 2026 URL split.** Depth and bookTicker are on `/public`; aggTrade, markPrice and forceOrder are on `/market`. Clients still on the old `/stream` URL keep receiving depth but silently lose the market channels.
- **Events recorded inline.** Connects, disconnects, stale feeds, gaps and resyncs are written to the same files, so the rebuild knows exactly where data cannot be trusted.
- **Clock offset.** Venue timestamps and local receive times come from different clocks. The server-time samples give an offset estimate (server time minus the midpoint of the request), so feed latency can be measured properly rather than as raw differences between two clocks.

### Layout on disk

```
data/
  raw/<venue>/<YYYY-MM-DD>/<HH>.ndjson.zst     one line per message: recv_ns<TAB>tag<TAB>payload
  status.json                                   live counters per feed
  logs/recorder.log
```

---

## Layer 2a: book rebuild

`python -m rebuild --start 2026-10-14 --end 2026-10-20` turns raw recordings into clean, replayable tables per venue, symbol and UTC day:

| Table | Contents |
|---|---|
| `deltas` | The order book as a replayable stream: level updates, resets (full book rebuilt from a snapshot) and invalid markers (book cannot be trusted until the next reset) |
| `top` | Best 10 levels after every valid book message |
| `trades` | Every trade, with aggressor side (+1 buyer, -1 seller) |
| `bbo` | Binance best bid and offer from bookTicker |

All sizes are in coins (OKX contract sizes are converted using the contract value from instrument metadata).

### How the books are validated

- **Binance**: synced with the documented procedure (REST snapshot, then events from the one containing `lastUpdateId`, then `pu` must equal the previous `u`). Every later REST snapshot is replayed forward through the same events and compared with the live book's top 20 levels. Because depth quantities are absolute, the two must match exactly. The report gives the match rate.
- **Bybit**: snapshot then deltas, with `u` required to advance by exactly one (measured on live data, not documented). A break invalidates the book until the next snapshot.
- **OKX**: `prevSeqId` must equal the previous `seqId`. The exchange's CRC32 checksum over the top 25 levels is also verified after every message and reported as a match rate.
- **All venues**: disconnects and stale feeds invalidate the book; crossed books after an update are counted.

A book is only written while it can be trusted. Each day file opens with a full reset, so any single day replays on its own. Processing starts two hours before the requested window so books are already in sync when it begins.

The tests generate a random true order book, encode it in each exchange's exact message format (including late REST snapshots, a dropped message, a disconnect and a midnight crossing), rebuild it, replay the output and compare it with the true book after every message.

---

## Setup on Windows

Open PowerShell in this folder.

```powershell
py -3.11 -m venv .venv          # or py -3.12 / py -3.13, any 3.11+
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q tests
```

### 1. Check the feeds (90 seconds)

```powershell
.venv\Scripts\python tools\check_feeds.py --seconds 90
```

This records briefly into `data\check`, then reports:
- message counts per channel
- clock offset and feed latency per venue
- the measured continuity rate for each venue's sequencing field
- a pass or fail list

If a venue fails, check `data\check\logs\recorder.log`. A geo-block shows up as an HTTP 451 or 403 error. Any venue can be switched off with `enabled = false` in `config.toml`.

### 2. Keep the PC awake

Run once, in an admin PowerShell:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
w32tm /resync
```

Also set Windows Update active hours, or pause updates for the recording period, so the PC does not restart overnight.

### 3. Run the recorder

Double-click `run_recorder.bat`, or:

```powershell
.\run_recorder.bat
```

It restarts automatically if Python exits. Stop it with Ctrl+C, then close the window. `data\status.json` updates every 15 seconds. The log prints a one-line summary per feed every minute.

### 4. Inspect what you have

```powershell
.venv\Scripts\python tools\inspect_raw.py
.venv\Scripts\python tools\inspect_raw.py --date 2026-10-13
```

Expected volume is roughly 3 GB per day compressed for all three venues.

### 5. Rebuild books

```powershell
.venv\Scripts\python -m rebuild --start 2026-10-14
```

Prints a quality report per venue and symbol (time in sync, resets, gaps, Binance snapshot cross-checks, OKX checksum match rate) and saves it to `data\clean\reports`. A full day takes roughly 20 to 30 minutes.
