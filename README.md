# QuantFlow

Generative market models for option-chain calibration and deep hedging, on equity index (SPX/VIX)
and crypto (BTC/ETH).

The goal is a **generative model of arbitrage-consistent market dynamics** — calibrated to real
option chains — first as a neural local-stochastic-volatility model, then via Schrödinger bridges — with **deep hedging** as the
downstream evaluation. `PROJECT_PLAN.md` is the design doc.

## Status

**Phase 1 of v1.0 is complete.** The data pipeline, the classical baseline, and the evaluation
harness that everything else will be measured through are all in place.

| | |
|---|---|
| **Data** | SPX option chains + price series via yfinance, cached as parquet |
| **Surface** | liquidity filters, Carr–Madan butterfly + calendar no-arbitrage filters, raw-SVI smile fit, Breeden–Litzenberger risk-neutral marginals |
| **Baseline** | Heston, calibrated by least-squares → **numpyro NUTS**, with full posterior diagnostics (R-hat, ESS, divergences) |
| **Simulator** | Heston Monte Carlo (**Andersen QE**), validated against the exact Fourier pricer |
| **Harness** | surface RMSE on a **held-out wing** of strikes, Wasserstein distance to the BL marginals, MC pricing error, martingale residual — all in `src/eval`, all reported with standard errors |

Results accumulate in [`reports/benchmark.md`](reports/benchmark.md).

Next: Phase 2 — a Dupire local-vol baseline and a neural local-stochastic-volatility model, scored through the same harness, in the same table.

## Quickstart

```bash
uv sync
uv run pytest                # 160+ tests, 100% coverage enforced
uv run ruff check src tests
```

Then read the notebooks in order:

- **[`notebooks/00_data_and_surface.ipynb`](notebooks/00_data_and_surface.ipynb)** — the SPX implied
  volatility surface and the recovered risk-neutral marginals.
- **[`notebooks/01_heston_mcmc.ipynb`](notebooks/01_heston_mcmc.ipynb)** — Heston calibrated by
  MCMC, with every benchmark number.

Both run offline from the committed parquet snapshots.

## Layout

```
src/
├── data/          # yfinance ETL, parquet cache
├── algorithms/    # black_scholes, preprocessing (SVI, BL), heston (Fourier + NUTS + QE MC)
└── eval/          # the shared harness: Paths, surface, metrics, benchmark, plots
notebooks/         # the narrative
reports/           # benchmark.csv (store) + benchmark.md (deliverable)
```

`src/eval` never imports `src/algorithms`: the harness must not be able to tell which model produced
the numbers it is scoring. Models depend on its `Paths` contract, not the other way round.

## License

Apache-2.0 for the library; CC-BY-SA 4.0 for the notebooks and writeups.
