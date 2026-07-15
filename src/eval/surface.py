"""Surface-level evaluation tooling: the train/test strike split and the IV mesh.

Numeric only -- plots.py draws what this returns, so the maths stays testable without a
display. Depends on SVIParams purely as a data carrier (it is a plain dataclass with a
total_variance method), never on a pricing model.
"""
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd

from ..algorithms.preprocessing.svi import SVIParams
from .eval_config import WING_BAND, WING_QUANTILE, SplitConfig

logger = getLogger()


@dataclass
class StrikeSplit:
    """A cleaned surface partitioned into a fitting set and a held-out wing."""

    train: pd.DataFrame
    test: pd.DataFrame


def split_wings(clean: pd.DataFrame, config: SplitConfig | None = None) -> StrikeSplit:
    """Hold out each maturity's extreme-moneyness quotes as an out-of-sample test set.

    PROJECT_PLAN.md §8 asks for "implied vol RMSE on held-out wing of strikes", and nothing in
    the pipeline provided it: every number reported so far is in-sample. The wings are where
    SVI extrapolates and where Heston's smile shape is most strained, so a model scored only
    on the strikes it was fitted to is grading its own homework.

    The split is per maturity, so every slice contributes to both sets -- a global split would
    hand the model whole maturities it never saw, which measures term-structure extrapolation
    (a different question) rather than wing fit.

    Slices too thin to survive the split (fewer than min_train quotes left to fit on)
    contribute nothing to the test set rather than being crippled.
    """
    cfg = config if config is not None else SplitConfig()
    if cfg.mode not in (WING_QUANTILE, WING_BAND):
        msg = f"Unknown split mode {cfg.mode!r}; expected {WING_QUANTILE!r} or {WING_BAND!r}."
        raise ValueError(msg)

    is_test = np.zeros(len(clean), dtype=bool)
    for ttm in clean["ttm"].unique():
        slice_mask = (clean["ttm"] == ttm).to_numpy()
        k = clean.loc[slice_mask, "log_moneyness"].to_numpy()

        if cfg.mode == WING_QUANTILE:
            # rank by DISTANCE from the money, so the held-out set is the wings on both
            # sides. A real SPX chain carries far more OTM puts than calls, so a symmetric
            # band would hold out a lopsided set; a quantile adapts to whatever coverage
            # the slice actually has.
            cut = np.quantile(np.abs(k), 1.0 - cfg.wing_frac)
            wing = np.abs(k) >= cut
        else:
            wing = (k < cfg.k_lo) | (k > cfg.k_hi)

        if slice_mask.sum() - wing.sum() < cfg.min_train:
            logger.warning("ttm=%.4f: only %d quotes, too thin to hold out a wing; "
                           "keeping the whole slice for training.", ttm, int(slice_mask.sum()))
            continue
        is_test[np.flatnonzero(slice_mask)[wing]] = True

    train = clean.loc[~is_test].reset_index(drop=True)
    test = clean.loc[is_test].reset_index(drop=True)
    logger.info("Wing split: %d train / %d test quotes across %d maturities.",
                len(train), len(test), clean["ttm"].nunique())

    return StrikeSplit(train=train, test=test)


def iv_surface_grid(svi: list[SVIParams], n_k: int = 60, n_t: int = 40,
                    k_lo: float = -0.30,
                    k_hi: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense (log-moneyness, maturity) implied-vol mesh from the fitted SVI slices.

    SVI is fitted per maturity, so there is no SVI in the time direction and the mesh has to
    interpolate between slices. It interpolates TOTAL VARIANCE w = sigma^2 * t linearly in t,
    then recovers sigma = sqrt(w / t). Interpolating implied vol directly would let w decrease
    across maturities, which is precisely calendar arbitrage -- the thing the preprocessor
    filters the raw quotes for. Doing it in w keeps the mesh arbitrage-free by construction.

    The mesh is in log-moneyness, not raw strike: the forward moves with maturity, so a
    raw-strike mesh smears the smile across the term structure.

    Returns (k_grid, t_grid, iv) with iv of shape (n_t, n_k).
    """
    if not svi:
        msg = "No SVI slices to build a surface from."
        raise ValueError(msg)

    slices = sorted(svi, key=lambda s: s.t)
    knots = np.array([s.t for s in slices])
    k_grid = np.linspace(k_lo, k_hi, n_k)
    t_grid = np.linspace(knots.min(), knots.max(), n_t)

    # (n_knots, n_k) total variance at the fitted maturities, then linear in t per k column
    w_knots = np.vstack([s.total_variance(k_grid) for s in slices])
    w_grid = np.vstack([np.interp(t_grid, knots, w_knots[:, j]) for j in range(n_k)]).T

    iv = np.sqrt(np.maximum(w_grid, 0.0) / t_grid[:, None])

    return k_grid, t_grid, iv


def svi_residuals(clean: pd.DataFrame, svi: list[SVIParams]) -> pd.DataFrame:
    """Per-quote SVI fit residuals (PROJECT_PLAN.md §8: "SVI fit residuals").

    SVI is the market *interpolant*, not a model under test, so its residual is the noise
    floor: no calibrated model should be expected to beat it, and a model that does is
    fitting quote noise. Reporting it alongside the Heston error is what makes the Heston
    number interpretable.

    Returns one row per quote in `clean` with iv_market, iv_svi and their difference.
    """
    rows = []
    for smile in svi:
        group = clean[np.isclose(clean["ttm"], smile.t)]
        iv_svi = smile.implied_vol(group["strike"].to_numpy())
        rows.append(pd.DataFrame({
            "ttm": group["ttm"].to_numpy(),
            "strike": group["strike"].to_numpy(),
            "log_moneyness": group["log_moneyness"].to_numpy(),
            "iv_market": group["iv"].to_numpy(),
            "iv_svi": iv_svi,
            "resid": iv_svi - group["iv"].to_numpy(),
        }))

    return pd.concat(rows, ignore_index=True)


def svi_iv_rmse(clean: pd.DataFrame, svi: list[SVIParams]) -> float:
    """RMSE of the SVI interpolant against the market IVs -- the model error's noise floor."""
    resid = svi_residuals(clean, svi)["resid"].to_numpy()
    finite = resid[np.isfinite(resid)]

    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")
