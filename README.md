# crypto-mm-research

Research stack for one question: **when the market's distribution shifts, does a model's uncertainty stay honest, and can a market maker use honest uncertainty to lose less to informed flow?**

This repository grows layer by layer:

1. **Recorder**: raw order book, trade, mark price, funding and liquidation data for BTC and ETH perpetuals on Binance, Bybit and OKX.
2. **Simulator**, in progress:
   - **Book rebuild** (done): validated order books and normalised trades from the raw data.
   - **Fill simulator** (done, Python reference version): queue position, latency, fees, markouts.
   - C++ port of the simulator core, checked against the Python version.
3. **Forecasting**, in progress:
   - **Features and walk-forward evaluation** (done).
   - **Conformal prediction intervals under distribution shift** (done).
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
- **OKX**: `prevSeqId` must equal the previous `seqId`. OKX documents a CRC32 checksum over the top 25 levels, but live data in October 2026 showed the public `books` channel sending `0` on every message, so OKX books are validated by sequence continuity alone. Binance books carry the stronger guarantee (exact match against exchange snapshots).
- **All venues**: disconnects and stale feeds invalidate the book; crossed books after an update are counted.

A book is only written while it can be trusted. Each day file opens with a full reset, so any single day replays on its own. Processing starts two hours before the requested window so books are already in sync when it begins.

The tests generate a random true order book, encode it in each exchange's exact message format (including late REST snapshots, a dropped message, a disconnect and a midnight crossing), rebuild it, replay the output and compare it with the true book after every message.

---

## Layer 2b: fill simulator

`python -m sim --date 2026-10-14` replays one day of a rebuilt book and runs a strategy against it, for every combination of fill model and latency.

**Clock and latency.** Everything runs on the exchange's clock. The strategy sees each book update `latency` after it happened, its orders and cancels reach the exchange `latency` after it sends them, and it hears about fills `latency` after they happen. Three settings: 150 ms (measured, recording from Glasgow), 10 ms (a server near the exchange) and 1 ms (colocated). A cancel can lose the race to a trade, exactly as it can live.

**Fill models** for a resting order at price P:

| Model | Filled when |
|---|---|
| pessimistic | the market trades through P, or the book crosses it |
| queue | the order joins behind the visible size at P. Trades at P consume the queue ahead first. When the level shrinks, traded volume is subtracted first (book updates already include trades, so counting them again would advance the order twice) and only the remainder counts as cancels, split between ahead and behind in proportion to ahead^n : behind^n (n = 1 by default) |
| optimistic | any trade at P, as if at the front of the queue |

**What the strategy sees.** On Binance the strategy is driven by the real-time best bid and offer (bookTicker), not the 100 ms depth stream. This matters: with only 100 ms batches, a 1 ms and a 10 ms strategy react to the same delayed picture, so latency below 100 ms cannot show up at all. The first run on real data showed exactly that (near-identical results at 1 ms and 10 ms). With the real-time feed, a fast quoter can pull a stale bid before a seller reaches it and a slow one cannot; `tests/test_sim.py::test_bbo_makes_latency_matter` builds that race explicitly. The real-time feed also sets the mid for markouts, triggers crossed-book fills at the exact moment, and gives the freshest size for queue position at the best price. Queue tracking below the best price uses the depth stream. `--no-bbo` runs the old depth-only mode for comparison.

Our orders are virtual and assumed small relative to the queues. Orders that would trade immediately are rejected (post-only). Orders are pulled while the book is invalid.

**Outputs** per run: fills, fill rate, time in queue, PnL before fees split into spread captured and inventory, markouts and adverse selection at 1, 5 and 30 seconds, and net PnL under three fee tiers (retail 2 bp, top VIP 0 bp, an illustrative -0.5 bp rebate).

The baseline strategy (`JoinQuoter`) quotes a fixed size at the best bid and ask and re-quotes whenever they move. It has no view on direction, so whatever it loses to informed traders shows up directly in the markouts. It is the benchmark the Layer 4 strategies must beat.

Tests in `tests/test_sim.py` are small enough to check on paper: queue arithmetic after trades and cancels, the double-counting trap, cancel splitting, model ranking, a cancel losing a race to a trade at 10 ms but winning at 1 ms, post-only rejection, invalid books, and the PnL split.

---

## Layer 3a: features and forecasts

`python -m forecast.features --dates 2026-10-14` builds one row every 100 ms for Binance BTCUSDT; `python -m forecast.evaluate --dates 2026-10-14,2026-10-15` runs the walk-forward test.

**One clock: ours.** Every feature at time t uses only messages our machine had received by t, and the targets are how the Binance mid we received moved over the next 1, 5 and 30 seconds. Exchange timestamps say when something happened, not when we could have known it. A cross-venue signal built on exchange time can look predictive purely because it uses news that had not reached us yet.

**Features**: best-level imbalance and microprice; 5-level depth imbalance; order flow imbalance at the best prices (Cont, Kukanov and Stoikov, 2014) over 1 s and 5 s; trade flow imbalance and trade count; past moves; spread; 30 s realised volatility; and for Bybit and OKX, their deviation from Binance relative to its 5-minute average, their recent move relative to Binance's, and their book imbalance. Rows where any book was out of sync or a feed was silent for 5 s are excluded.

**Evaluation**: data is cut into 1-hour blocks; each block is predicted by models trained only on earlier blocks, with a 30-second gap so no training target overlaps the test period. Nothing is shuffled. R² is measured against predicting no change (not the sample mean, which a live system cannot know). Models: ridge regression on standardised features, and small, heavily regularised gradient-boosted trees (LightGBM). Rows 100 ms apart with targets up to 30 s long are far from independent, so results are treated as estimates over hours, not over hundreds of thousands of samples.

**Tests** (`tests/test_forecast.py`):
- no look-ahead: everything received after time T is altered, and no feature at or before T, nor any target ending before T, may change
- signal recovery: in a synthetic market where Binance follows OKX by 2 s, the OKX feature and the out-of-sample model must find it
- negative control: in a pure random walk, out-of-sample skill must be about zero
- order flow imbalance against a hand-worked example

---

## Layer 3b: intervals that hold when the market shifts

`python -m forecast.intervals --dates 2026-10-14` wraps every out-of-sample forecast in a 90% interval using five methods and checks whether each really contains the outcome 90% of the time: overall, in calm and stressed markets, in the most volatile 5% of moments, and in every 10-minute window.

| Method | How it calibrates |
|---|---|
| static | split conformal on the first hour of out-of-sample residuals, then frozen |
| rolling | recalibrated every second on the trailing 10 minutes |
| scaled | as rolling, on residuals divided by recent volatility, so intervals widen and narrow with the market |
| regime | scaled, calibrated separately in calm, normal and stressed volatility terciles (Mondrian conformal) |
| adaptive | adaptive conformal inference (Gibbs and Candès, 2021): the working miscoverage level is nudged after every update, alpha_t+1 = alpha_t + gamma (alpha - recent miss rate) |

**Label delay.** A 30-second forecast cannot be scored until 30 seconds later. Every calibration set at time u uses only rows whose outcomes had arrived by u, and the adaptive update counts only misses already known. Quantiles use the finite-sample rank ceil((n + 1)(1 - alpha)).

**Origin.** The scaled and regime methods carry over two ideas from my MSc work on turbulence surrogates: absolute versus scaled conformal intervals, and calibrating separately per flow region. Markets add what the turbulence data never had: a distribution that shifts while the model is running.

**Tests** (`tests/test_conformal.py`), on synthetic residuals with known answers:
- with nothing changing, every method covers 90% (within 2%)
- noise quadruples and volatility shows it: static collapses below 50%; scaled, regime and adaptive stay at 90%
- noise triples but volatility does not show it (an unobserved shift): in minutes 1 to 3 after the shift, scaled covers about 56% while adaptive is back to about 92%; the window methods only recover after about 10 minutes; static never does
- no look-ahead: changing outcomes that arrive after time T changes no interval set before T

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

Prints a quality report per venue and symbol (time in sync, resets, gaps, Binance snapshot cross-checks) and saves it to `data\clean\reports`. A full day takes roughly 20 to 30 minutes.

### 6. Run the simulator

```powershell
.venv\Scripts\python -m sim --date 2026-10-14
.venv\Scripts\python -m sim --date 2026-10-14 --hours 2      # quick run on the first 2 hours
```

Runs the baseline quoter on Binance BTCUSDT for all three fill models and all three latencies, prints a summary table and saves full results to `data\results`. A full day takes roughly 10 minutes.
