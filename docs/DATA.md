# Data reference

What is stored on disk, what every column means, and what preprocessing turns it into. Written
so that coming back to the project after a break does not require re-reading the ETL.

The authoritative code is `src/data/data_loader.py` (ETL), `scripts/snapshot_chain.py` (dated
snapshots) and `src/algorithms/preprocessing/` (everything downstream). This file describes
them; if the two disagree, the code wins and this file is stale.

## Files on disk

`data/` is gitignored except for the two committed snapshots the notebooks run from.

| path | rows | contents |
|---|---|---|
| `data/spx_option_chain.parquet` | 2 671 | **the frozen Phase 1 snapshot** — SPX chain as of 2026-06-16 20:01:05 UTC, spot 7511.58 |
| `data/chains/spx_option_chain_<YYYY-MM-DD>.parquet` | ~2 100 each | the daily accumulating series, identical schema |
| `data/spx_price.parquet` | 2 628 | SPX daily OHLCV, 2016-01-04 → present |

**The frozen snapshot is frozen.** Every number in `reports/benchmark.{csv,md}` is calibrated to
`data/spx_option_chain.parquet`; regenerating it silently invalidates the whole table.
`scripts/snapshot_chain.py` never writes to it — dated captures go to `data/chains/` only.

Why a daily capture exists at all: Yahoo serves only the *live* chain. `option_chain(date=...)`
selects an expiry, not an as-of date, and there is no history behind it, so a trading day that is
not captured is lost permanently (PROJECT_PLAN §7). The script is idempotent (skips a date already
on disk) and refuses to write weekend/holiday replays, which it detects by checking that at least
one contract's `lastTradeDate` falls on the snapshot date.

## Option-chain schema (19 columns)

One row per quoted contract, calls and puts, across the whole maturity ladder.

### Straight from yfinance (13)

| column | dtype | notes |
|---|---|---|
| `contractSymbol` | str | e.g. `SPXW260623C06600000` — SPXW = weekly, C/P, then strike ×1000 |
| `lastTradeDate` | datetime64[ms, UTC] | when this contract last traded; can be weeks stale |
| `strike` | float64 | |
| `lastPrice` | float64 | last trade. Stale for illiquid strikes — used only as a mid fallback |
| `bid`, `ask` | float64 | 0.0 means no market, not a zero price |
| `change`, `percentChange` | float64 | almost always 0.0 in these snapshots; unused |
| `volume` | float64 | **NaN when the contract has not traded today**, not 0 |
| `openInterest` | float64 / int64 | dtype varies by snapshot (NaN present ⇒ float) |
| `impliedVolatility` | float64 | Yahoo's own IV. **Not used** — the preprocessor recomputes IV from the mid with its own forward and rate |
| `inTheMoney` | bool | relative to spot, not the forward |
| `contractSize`, `currency` | str | constant `REGULAR` / `USD` |

### Added by `DataLoader.load_option_chain` (6)

| column | dtype | notes |
|---|---|---|
| `expiry` | str | `YYYY-MM-DD`, the yfinance expiry key |
| `type` | str | `"call"` or `"put"` |
| `spot` | float64 | constant across the frame; `regularMarketPrice` from Yahoo's underlying quote |
| `snapshot_time` | datetime64[ms, UTC] | constant; `regularMarketTime`, the instant the quotes were taken |
| `ttm` | float64 | years to expiry, `(expiry − snapshot_time) / 365d`, measured from the snapshot **not the wall clock** |

`spot` and `snapshot_time` are captured once, from the first expiry's underlying quote, and
broadcast — so they are consistent across the frame rather than drifting between HTTP calls.

The point of the six added columns is that the frame is **self-contained**: `s_0`, the valuation
date and `t` all live in it, so a calibration never needs to join against the price file.

### The maturity ladder

`_select_maturity_ladder` picks up to `n_maturities=8` expiries spread evenly over
`[min_days=7, max_days=400]`, rather than the nearest 8. For SPX the nearest dozen expiries are
daily/weekly contracts inside a two-week window, and Heston's `kappa`/`theta` are only identifiable
across a term structure.

The frozen snapshot's ladder:

| expiry | quotes | ttm (yr) | strike range |
|---|---|---|---|
| 2026-06-23 | 355 | 0.0169 | 3000 – 9600 |
| 2026-07-01 | 292 | 0.0388 | 3000 – 9800 |
| 2026-07-10 | 329 | 0.0635 | 2400 – 9600 |
| 2026-07-20 | 107 | 0.0909 | 3000 – 9400 |
| 2026-08-07 | 188 | 0.1402 | 3000 – 9800 |
| 2026-10-16 | 608 | 0.3320 | 200 – 12400 |
| 2027-01-15 | 383 | 0.5813 | 200 – 11400 |
| 2027-06-17 | 409 | 1.0005 | 200 – 11400 |

Split 1 186 calls / 1 485 puts. Note the very short front maturity (0.0169 ≈ 6 days) — that is the
slice that strains the Fourier pricer's `u_max`, see CLAUDE.md.

## Price schema

`Close, High, Low, Open, Volume` on a `DatetimeIndex` named `Date`. Volume is `int64`, the rest
`float64`. No adjusted-close column: `^SPX` is an index, so there is nothing to adjust.

**Load it through `DataLoader.load_spx`, not `pd.read_parquet`.** yfinance hands back MultiIndex
`(field, ticker)` columns on a `DatetimeIndex`; `to_parquet` rejects MultiIndex columns outright,
so `save_parquet` resets the index to a plain `Date` column and `_normalise_price_frame` restores
it on the way back in. Read the file raw and you get a `RangeIndex` with a stray `Date` column.

`load_spx(start, end, interval, path=...)` prefers the cache and falls back to yfinance on a miss
(no file, or an unreadable one), persisting the download — so re-runs are reproducible offline.
Only file-level read errors are swallowed; anything else propagates rather than turning into a
surprise network call.

## What preprocessing turns it into

The rate is the one input that does not come from the chain's own columns, and it is not in the
yfinance feed. `implied_rate(chain)` backs it out of put-call parity — regressing the call−put mid
spread on strike gives `−e^{−rT}` as the slope, with dividends already folded into the forward —
and takes the median across expiries (a single crossed quote should not move the whole surface).
On the frozen snapshot: **r = 0.0573**. `implied_rate_curve(chain)` returns the per-expiry version
as `(ttm, rate, n_quotes)`; the *spread* of that curve is worth a look whenever the BL marginals
fail to recover the forward.

```python
chain = DataLoader().load_parquet("data/spx_option_chain.parquet")
r = implied_rate(chain)
surface = SurfacePreprocessor().prepare(chain, r)   # -> PreparedSurface
```

`PreparedSurface` has three fields.

### `clean` — a pandas frame, and a *different* one

Not a filtered chain: one row per surviving **OTM call-equivalent**, puts folded in by parity.

| column | meaning |
|---|---|
| `ttm` | maturity in years |
| `strike` | |
| `forward` | `spot · e^{rT}`, constant within a maturity |
| `log_moneyness` | `log(K / F)` |
| `call_price` | undiscounted-equivalent call mid; puts converted via `P + e^{−rT}(F − K)` |
| `iv` | Black–Scholes implied vol, recomputed from `call_price` |
| `total_var` | `iv² · t` |

Attrition on the frozen snapshot is **2 671 → 1 045 rows**, in this order: liquidity filter
(`volume ≥ 1`, `openInterest ≥ 1`, relative spread ≤ 0.5, non-zero bid), OTM-only selection,
IV inversion (quotes outside the no-arbitrage bounds fail to invert and are dropped with a
warning), Carr–Madan butterfly convexity, then the calendar filter across maturities. Per
maturity: 133, 109, 122, 44, 91, 224, 153, 169.

### `svi` — `list[SVIParams]`, one per maturity

`a, b, rho, m, sigma` (raw SVI, `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`) plus the `t` and
`forward` the slice was fitted at. Fitted by `scipy.least_squares` with vega weighting, so the
fit is effectively in IV space rather than total-variance space.

### `marginals` — `list[Marginal]`, one per maturity

`ttm`, `strikes` (400-point dense grid), `density`, `raw_mass`. Breeden–Litzenberger applied to
the fitted SVI smile.

`raw_mass` is the density's integral **before** renormalisation, and it exists to keep the
truncation honest: the grid spans the traded strikes only, so probability outside it is not
measured. On the frozen snapshot it sits within ±0.6% of 1 at every maturity except the 0.58y
slice (1.054).

**Do not widen the BL grid.** It is `grid_mode="traded"` by default for a reason documented at
length in `MarginalConfig`: past the last traded strike, BL is reading curvature off SVI's
*extrapolated* wing, which invents tail mass, and since `E[S_T] = ∫K μ(K) dK` weights by K that
fake mass dominates. A grid to 1.5F or 3F pushes forward-recovery error to +23–27%; the traded
grid holds it under 1%. `grid_mode="fixed"` exists for synthetic surfaces, where the smile *is*
the true model and extrapolating is safe.

## Related: the simulated-path contract

Model *output* — as opposed to market data — travels as `src/eval/paths.py:Paths`
(`times`, `spot`, `variance`, `r`), numpy-only and deliberately model-agnostic. That is the
contract the evaluation harness scores; see CLAUDE.md's architectural rule.
