"""Hyperparameter configs for the evaluation harness.

Mirrors the heston_config.py / preprocessing_config.py pattern: small dataclasses, sensible
v1 defaults, bundled where it helps.
"""
from dataclasses import dataclass

WING_QUANTILE = "quantile"
WING_BAND = "band"


@dataclass
class SplitConfig:
    """How to hold out a wing of strikes for out-of-sample scoring.

    PROJECT_PLAN.md §8 asks for "implied vol RMSE on held-out wing of strikes". The wings are
    where SVI extrapolates and where Heston's smile shape is most strained, so a fit scored
    only in-sample flatters itself.

    mode="quantile" (default): per maturity, rank quotes by |log_moneyness| and hold out the
    top `wing_frac`. Robust to the wildly asymmetric strike coverage of a real SPX chain,
    which carries far more OTM puts than OTM calls.
    mode="band": hold out everything outside [k_lo, k_hi] in log-moneyness -- the economically
    literal "wing", but it can empty a thin slice entirely.

    min_train guarantees no maturity is left with too few quotes to calibrate on; a slice that
    would fall below it contributes nothing to the test set instead.
    """

    mode: str = WING_QUANTILE
    wing_frac: float = 0.25      # quantile mode: fraction of each slice held out
    k_lo: float = -0.15          # band mode: log-moneyness band to keep
    k_hi: float = 0.10
    min_train: int = 5           # never shrink a maturity's training slice below this


@dataclass
class BenchmarkConfig:
    """Where the growing results table lives.

    The csv is the store and the markdown is rendered from it -- parsing markdown back into a
    DataFrame to achieve idempotent updates is a bug farm (float round-trip drift, NaN
    rendering, escaping), and pandas' own to_markdown needs `tabulate`, which is not a project
    dependency.
    """

    csv_path: str = "reports/benchmark.csv"
    md_path: str = "reports/benchmark.md"
