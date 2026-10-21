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
4. **Uncertainty-aware market maker** (done, ready for real data): forecast- and interval-driven quoting, compared against the naive quoter in the simulator.
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

**Adaptive safeguards, and the bug that led to them.** The first real-data run exposed a fault: when a burst of misses pushed the working level alpha_t to zero or below, the interval became infinite, and infinite intervals were left out of the update, so the method stopped learning and stayed infinite for most of the day. Gibbs and Candès count an infinite interval as covering, which lets the method recover. The fix does that, keeps alpha_t within [0, 0.5] so the interval is capped at the widest calibration residual rather than infinite, and scales the step size by 1/sqrt(horizon) because feedback on longer forecasts arrives later and misses come in correlated runs. Evaluation now counts infinite intervals as covering and reports their share, so no method can look better by dropping its hardest moments. `tests/test_conformal.py::test_adaptive_never_stalls_after_a_burst_of_misses` reproduces the failure.

**Label delay.** A 30-second forecast cannot be scored until 30 seconds later. Every calibration set at time u uses only rows whose outcomes had arrived by u, and the adaptive update counts only misses already known. Quantiles use the finite-sample rank ceil((n + 1)(1 - alpha)).

**Origin.** The scaled and regime methods carry over two ideas from my MSc work on turbulence surrogates: absolute versus scaled conformal intervals, and calibrating separately per flow region. Markets add what the turbulence data never had: a distribution that shifts while the model is running.

**Tests** (`tests/test_conformal.py`), on synthetic residuals with known answers:
- with nothing changing, every method covers 90% (within 2%)
- noise quadruples and volatility shows it: static collapses below 50%; scaled, regime and adaptive stay at 90%
- noise triples but volatility does not show it (an unobserved shift): in minutes 1 to 3 after the shift, scaled covers about 56% while adaptive is back to about 92%; the window methods only recover after about 10 minutes; static never does
- no look-ahead: changing outcomes that arrive after time T changes no interval set before T

---

## Layer 4: quoting with forecasts and their uncertainty

`python -m sim.layer4 --date 2026-10-14` runs four strategies through the fill simulator over the same hours, at each latency, using the forecasts and intervals from Layer 3.

| Strategy | Rule |
|---|---|
| join | quote at the best bid and ask (the baseline) |
| forecast | quote one side only when the forecast is strong relative to its uncertainty, \|f\| > 0.25 w: a falling forecast pulls the bid, a rising one pulls the ask |
| width | step back from the best price by 5 ticks per unit of rho above 1, where rho = interval width / its 10-minute typical value; withdraw both sides when rho > 2.5 |
| full | forecast and width together |

All except join also lean against inventory (2 ticks per full position limit), and quotes never cross the spread.

Because of that, the first comparison mixes two changes: a forecast variant differs from join both by its own rule and by the inventory lean. Three ablations separate them: `join_skew` (join plus the lean), `forecast_noskew` and `width_noskew`. The table also reports `requotes_per_fill`: every cancel to move or pull a quote sends it to the back of the queue, and fills taken from the back of the queue are the ones that arrive just before the level is cleared, which are the most adversely selected. First real-data run (15 October, ridge 5 s forecasts, adaptive intervals): every forecast and width variant had fewer fills but worse adverse selection per fill than join, at all three latencies. The ablations test whether that comes from the signal or from losing queue position.

Ablation result (same day, 150 ms): the inventory lean was the main cause. `join_skew` lost $273 more than join, and `forecast_noskew` was the best strategy in the table (gross -$1,116 against join's -$1,290; +$98 against +$36 with the rebate). Requotes per fill were similar for every strategy (11 to 13), so lost queue position from re-quoting was not the main effect. The width rule still hurt without the lean. Both the lean and the width rule work by stepping a quote back from the best price, and on a one-tick-spread market a quote behind the best price only fills when the price moves through it. So the next variants control risk by pulling a side instead (`join_pull`, `forecast_pull`, `width_pull`, `full_pull`), and every strategy is now compared with join hour by hour (`vs_ref_usd_per_h`, `t_vs_ref`, against join unless `--reference` names another strategy) so a single day's difference is not mistaken for an edge.

**Clock alignment without look-ahead.** The simulator runs on the exchange clock, the forecasts on our receive clock. A strategy at simulator time t has seen Binance up to exchange time t - latency; the signal finds when our machine received that update and uses the latest forecast made by then. Limitation: forecasts reflect what reached Glasgow, so in the 10 ms and 1 ms scenarios only order timing improves, not the forecast's information.

**Tests** (`tests/test_layer4.py`): the signal never returns a forecast made after the information the strategy has seen; every quoting rule (neutral, one-sided, stepping back, withdrawing, inventory lean, never crossing, position limits); and an end-to-end market where an informed trader sweeps one side 50 ms before every move: the join quoter is caught on all 200 moves, the forecast quoter with a perfect forecast on none.

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


### Pull result and the control it needs

Same day, 150 ms. Every pull variant beat join hour by hour: `full_pull` lost $849 against join's $1,290 (t = 8.5 over 16 hours, +$113 with the rebate). But every strategy here loses about 0.45 bp per fill before rebates, so any rule that trades less loses less. `full_pull` made 27% fewer fills; join simply thinned by 27% would have lost about $940. The honest comparison is therefore against join with random withdrawals at the same fill count (`random_off`, rate set with `--random-off`), and per fill (`pnl_gross_bps`, `adverse_*`), not in total dollars. No strategy is profitable before the rebate, and none at a retail fee.

Control result (15 October, 150 ms, in-sample). Join withdrawing at random 30% of the time with the same inventory pull (`random_off_pull`) lost $863; `full_pull` lost $849 while making 2,000 more fills. Per fill, `full_pull` was better (gross -0.44 bp against -0.48, adverse at 1 s -0.37 against -0.39). So most of the improvement over join came from three things that need no forecast: not stepping back from the best price, pulling a side to control inventory, and trading less. The forecast and its uncertainty add a smaller improvement in fill quality, which still has to hold up on a day the settings never saw.

Fill-matched control (15 October, `--random-off 0.25`, reference `random_off_pull`): 27,535 fills against `full_pull`'s 27,732. `full_pull` lost $849 against $940 before rebates (gross -0.44 bp per fill against -0.49; adverse at 1 s -0.37 against -0.40), +$4.5 per hour on the 5 s markout with t = 1.8 over the day's hours. Suggestive, not conclusive, and in-sample.

### Frozen for the out-of-sample test

Fixed on 16 October 2026 before any data from that day was used, and committed so the timestamp shows it. Nothing below may change after seeing the result.

* Forecast: ridge, 5 s horizon, features and walk-forward settings as in Layer 3.
* Intervals: adaptive, 90%, gamma 0.005, 10-minute window.
* Strategies: `join`, `random_off_pull` (random withdrawal 0.25), `full_pull` (theta 0.25, rho_max 1.5, pull_at 0.4, no lean, no stepping back).
* Latency 150 ms, queue fill model, size 0.01 BTC, position limit 0.05 BTC.
* Test: `full_pull` against `random_off_pull`, hour by hour (`t_vs_ref`), and per fill (`pnl_gross_bps`, `adverse_1s_bps`).

```
python -m sim.layer4 --date 2026-10-16 --intervals-dates 2026-10-14,2026-10-16 --latency-ms 150 --strategies join,random_off_pull,full_pull --random-off 0.25 --reference random_off_pull --label oos
```

The out-of-sample test is run on every full day recorded after the freeze (16 October onwards), one `sim.layer4 ... --label oos` run per day, then pooled with `python tools/pool_oos.py`: the hourly comparison over every hour of every day, the per-fill numbers over all fills, and the number of days on which `full_pull` was better per fill.

### Out-of-sample result (16 to 18 October, frozen settings)

`full_pull` against `random_off_pull` (25% withdrawal), 150 ms: better per unit traded on 3 of 3 days (pooled -0.342 bp against -0.404), lower adverse selection every day, hourly dollar comparison t = 9.3 over 63 hours. The fill counts did not stay matched out of sample (`full_pull` traded 15 to 40% less than the control), and because every strategy loses per fill, trading less flatters the dollar comparison. Two stricter checks were therefore added after seeing these results. They change no strategy setting; they only make the comparison harder to pass:

* `--match full_pull` sets the control's withdrawal rate each day, by bisection, so its fill count is within 1% of `full_pull`'s.
* `vs_ref_bps_per_fill` and `t_per_fill` compare 5 s markout per unit traded, hour by hour, which trading less cannot improve.

Run with `--label oos_matched` and pooled with `python tools/pool_oos.py --label oos_matched`.

### Volatility-filter control (stated before running, 20 October 2026)

`full_pull` withdraws when its interval is unusually wide, and the adaptive interval is scaled by recent volatility, so part of its gain could come from stepping away in volatile moments, which needs no forecast. `vol_pull` tests that: the same inventory rule, plus withdrawal of both sides whenever rv_30s is above `vol_max` times its typical level, with `vol_max` set each day by bisection so its fill count is within 1% of `full_pull`'s. Same three test days, same settings for `full_pull`.

Reading, fixed in advance: pooled `t_per_fill` of `full_pull` against `vol_pull` above 2 means the forecasts and intervals add something beyond volatility timing; below 2 means the out-of-sample gain is explained by when volatility was high, and the note will say so.

```
python -m sim.layer4 --date 2026-10-16 --intervals-dates 2026-10-14,2026-10-18 --latency-ms 150 --strategies vol_pull,full_pull --reference vol_pull --match full_pull --label volctl
python tools/pool_oos.py --label volctl --ref vol_pull
```

## Version 2: conformal market making with an adverse-selection budget

`ConformalBudgetQuoter` (in `sim/strategies.py`) replaces the three hand-set rules of `full_pull` with one rule and one dial.

* Score per side = the interval's worst-case adverse move against that side: `w - f` for the bid, `w + f` for the ask (bp). It combines the forecast and its uncertainty with no weights to choose.
* A side is quoted only while its score is at or below a threshold `lam`.
* `lam` is tuned online so the realised 5 s markout loss of the strategy's own fills stays at a budget `B` (bp per unit traded): each fill is settled 5 s later from the strategy's own view of the book, then `lam <- lam - gamma * (loss - B)`. This is the adaptive conformal inference update applied to a loss (online conformal risk control). With losses clipped to [-20, 20] bp and `lam` inside [0.5, 10], the average settled loss over N fills is within (10 - 0.5) / (gamma N) of `B` whatever the market does; the run reports how often `lam` sat on a bound, where that guarantee does not hold.
* Inventory: the same pull rule as `full_pull`.

`python -m sim.frontier` sweeps each method's one dial (random withdrawal rate, volatility threshold, budget `B`) and saves fill volume against adverse selection for every setting, so methods are compared as whole curves rather than at one setting each. `python tools/pool_frontier.py --dates ...` pools days.

### Plan, stated before any real-data run (20 October 2026)

Development days: 15 to 18 October (already used for version 1). `gamma` = 0.02 and the budget grid {0.1, 0.2, 0.3, 0.4} bp are fixed now and will not change after seeing development results, unless a run fails outright (for example `lam` stuck on a bound for most updates), in which case the change and its reason go here before any test-day run.

Test days: the first complete days recorded after the recorder restart on 20 October (21 October onwards), pooled.

Claims and how they are read:

1. Budget guarantee: for every budget where `lam` sat on a bound for under 5% of updates, the pooled realised 5 s markout loss is within 0.05 bp of `B`.
2. Frontier: at the fill volume of each conformal point, its 5 s markout per unit traded is better than the volatility filter's, interpolated along the volatility-filter curve. Reported for every budget, not only the best.

### Development result and one amendment (20 October 2026, before any test day)

Pooled over 15 to 18 October (development days):

| Method | Notional ($m) | 5 s markout loss per unit (bp) |
| --- | --- | --- |
| Conformal, B = 0.1 / 0.2 / 0.3 / 0.4 | 5.2 / 11.2 / 25.1 / 46.0 | 0.180 / 0.219 / 0.286 / 0.359 |
| Volatility filter, threshold 0.8 to 2.0 | 18.8 to 58.5 | 0.378 to 0.391 |
| Random withdrawal, rate 0.8 to 0.2 | 13.9 to 53.0 | 0.442 to 0.400 |
| full_pull (version 1) | 43.1 | 0.367 |
| join | 69.6 | 0.406 |

The volatility filter and random withdrawal stay near 0.38 to 0.44 bp however little they trade; the conformal quoter's loss falls as its budget tightens, and at matched volume it is below the volatility filter at every budget.

Budget check: realised loss 0.178 / 0.221 / 0.294 / 0.363 bp against budgets 0.1 / 0.2 / 0.3 / 0.4. The miss at B = 0.1 is the start-up transient the guarantee allows for: each day started at lam = 3.0, far above where B = 0.1 settles (about 0.65), and with few fills per day the bound (lam_start - lam_end) / (gamma N) is about 0.06 bp. At B = 0.4 on weekend days lam reached its upper bound in 13 to 14% of updates (the strategy was below budget even quoting almost everything), so that guarantee did not apply there.

Amendment, made now, before any test-day data is processed: a live strategy runs continuously, so the threshold should carry over from one day to the next. `--lam0-from <date>` starts each budget at its final threshold from an earlier frontier file. On the test days both versions are reported: cold start (as pre-registered) and warm start (amended). Claim 1 is judged on the warm start; the cold-start numbers are shown beside it.

## Market studies (research/)

Studies of how this market works, using the same recordings, each feeding back into the strategy.

`python -m research.events --dates ...` extracts liquidations (all three venues), mark price and funding, and the per-minute clock samples from the raw files into `data/clean/events/<date>/`. Side is normalised to the forced order: +1 buys (a short was liquidated), -1 sells. Bybit reports the liquidated position's side, so it is inverted; OKX sizes are contracts (0.01 BTC); Binance pushes at most one liquidation per symbol per second, so its counts are a lower bound.

`python -m research.liquidations --dates ...` groups BTC liquidations across venues into episodes and measures the Binance mid path from 30 s before to 120 s after, signed by the forced direction; compares forced flow with voluntary aggressive flow of the same size (net aggressive Binance notional in the first 2 s, controls at least 60 s from any liquidation); checks each venue's side convention against its own trades; and compares how fast each venue's mid moves.

`python -m research.queue --dates ... --intervals-dates ...` splits every simulated fill by how it happened (our turn in the queue, a trade through our price, or the book crossing us), by the volume ahead of us when we joined, by time in the queue, and by whether forced flow was trading against us in the 10 s after a liquidation.

Expectations written before running on real data (20 October 2026):

1. Side conventions: at least 80% of liquidations match the aggressor side of their venue's trades within 200 ms.
2. Forced flow is less informed than voluntary flow of the same size: a smaller share of its early move is still there after 120 s.
3. Fills from a trade through our price or the book crossing us are far more adversely selected than fills where our turn in the queue came.
4. Fills where forced flow traded against us are the most adversely selected group.

### Market study results, development days 15 to 18 October

Against the expectations stated before running:

1. Side conventions: 76% (Binance), 74% (Bybit) and 74% (OKX) of liquidations match the aggressor side of their venue's trades within 200 ms. Below the 80% I expected, but far from 50%, and a wrong Bybit inversion would have shown about 26%. The conventions are right; the 200 ms window also catches ordinary trades on the other side.
2. Wrong. Forced flow does not fade more than voluntary flow. Its first-second impact is smaller (largest tercile +0.29 bp against +0.55 bp) but it keeps drifting in the forced direction (+1.36 bp at 120 s against +0.88 bp), while voluntary flow partly reverses. The bigger finding is before the liquidation: the Binance mid had already moved 4.85 bp in the forced direction over the previous 30 s (7.3 bp for the largest tercile, against about 0.3 bp before voluntary flow). Liquidations arrive at the end of a move they did not start, then the move keeps creeping on for minutes, consistent with one liquidation triggering the next.
3. Confirmed. Fills from a trade through our price or the book crossing us are 9% of join_pull's fills and lose 0.79 to 0.88 bp at 5 s, against 0.34 bp for fills where our turn in the queue came; they carry about a fifth of all losses. Queue position matters more than anything else measured: fills that joined with almost nothing ahead (front quintile, under 0.005 BTC) broke even at 5 s and made +0.10 bp at 30 s before fees; the back quintile (over about 1 BTC ahead) lost 0.68 bp.
4. Partly confirmed. Fills in the 10 s after a liquidation episode where the forced flow traded against us are no worse at 1 s or 5 s, but lose 0.68 bp (join_pull) and 1.09 bp (full_pull) at 30 s, against about 0.42 and 0.37 bp outside liquidations. The strategies forecast 5 s ahead, so they cannot see this.

Not yet interpreted: the time for each venue's mid to make half its 5 s move after an episode (Bybit 0.95 s, OKX 1.05 s, Binance 1.20 s). The Binance mid is the real-time best bid and offer, the others are from depth updates, and exchange clocks differ, so this needs the proper lead-lag study before any claim.

## Version 3 and the lead-lag study (21 October 2026)

`V3Quoter` keeps version 2's budget rule and adds what the market studies found, each switchable so it can be ablated:

* queue: a new order's score includes the measured cost of its queue position, 0.1 bp per e-fold of visible volume ahead above 0.002 BTC (fitted by eye to the development-day queue study: 0 bp near the front, about 0.68 bp behind 1 BTC). An order already resting at the price keeps its place.
* liq: while net forced notional over the last 60 s is at least $100,000 in one direction, the side that flow would hit is not quoted.

Frontier variants: `conformal_q` (queue only), `conformal_l` (liquidations only), `conformal_ql` (both: version 3), each over budgets 0.1 to 0.4 bp. Run with `--only conformal,conformal_q,conformal_l,conformal_ql` on development days first. The constants above (0.1, 0.002 BTC, 60 s, $100,000) come from the development studies; after the development run they are frozen and the version 3 test plan is written here before any test day is used.

`python -m research.leadlag --dates ...` measures which venue's mid moves first, by hourly cross-correlation of 10 ms returns at lags up to 2 s, on both exchange timestamps (who moves first at the source) and receive timestamps (whose data reaches Glasgow first), with each exchange's clock offset and round trip from the per-minute clock samples.

### Version 3 development result and freeze (21 October 2026, before any data from 22 October is processed)

Pooled over 15 to 18 October, notional traded and 5 s markout loss per unit traded (30 s in brackets):

| Budget B | version 2 | + queue cost | + liquidation guard | both |
| --- | --- | --- | --- | --- |
| 0.1 | $5.2m, 0.180 (0.182) | $8.2m, 0.134 (0.123) | $5.0m, 0.183 (0.204) | $8.1m, 0.133 (0.132) |
| 0.2 | $11.2m, 0.219 (0.222) | $16.7m, 0.205 (0.217) | $11.5m, 0.218 (0.241) | $15.9m, 0.205 (0.228) |
| 0.3 | $25.1m, 0.286 (0.327) | $29.4m, 0.286 (0.328) | $24.0m, 0.286 (0.323) | $28.2m, 0.286 (0.313) |
| 0.4 | $46.0m, 0.359 (0.390) | $47.5m, 0.356 (0.379) | $45.4m, 0.357 (0.387) | $45.4m, 0.354 (0.378) |

* The queue cost works where the strategy is selective: at B = 0.1 it traded 58% more and lost a quarter less per unit; at about $11m of volume its loss is about 0.16 bp against 0.22 bp for version 2. At B = 0.3 and 0.4 the threshold rarely binds and it changes little.
* The liquidation guard did nothing measurable, at 5 s or at 30 s. Two reasons, both worth knowing: liquidations arrive after a large move, when the volatility-scaled interval is already wide and version 2 has already pulled back; and under a budget, any filter that removes bad fills raises the threshold and buys other fills back. Recorded as a negative result and dropped.

Frozen as version 3: `conformal_q` (version 2 plus the queue cost, q_slope 0.1, q_ref 0.002 BTC), gamma 0.02, budgets {0.1, 0.2, 0.3, 0.4}.

Test, on the same test days as version 2 (22 and 23 October, the first complete days after the recorder restart, not yet processed), pooled, warm start from the development run (`--lam0-from 2026-10-18_v3dev`) with the cold start shown beside it:

1. At the notional of each version 3 point with B = 0.1 and 0.2, its 5 s markout loss per unit is below version 2's, interpolated along version 2's curve.
2. The budget check from version 2 applies unchanged.

### Lead-lag, first run

On receive timestamps (book mids), Binance moves first: Bybit and OKX follow about 30 ms later, in 99 to 100% of hours. Bybit against OKX shows spikes every 0.1 s, which is their 100 ms book publication cadence rather than price discovery, so book mids cannot settle sub-100 ms leads for those two. The exchange-clock section came back empty because of a bug (each day's files hold a few rows from before midnight, and the day was taken from the first row); fixed, and the study now also runs on trade prices, which every venue publishes trade by trade. Clock samples: every exchange's clock reads 35 to 53 ms ahead of this laptop, consistently, so the laptop's clock is behind; the differences between exchanges (up to 18 ms) are within what round-trip asymmetry over a 250 to 330 ms round trip can produce, so they are not interpreted.
