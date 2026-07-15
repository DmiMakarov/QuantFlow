"""The Phase-1 metrics: calibration accuracy, distributional fidelity, no-arbitrage.

PROJECT_PLAN.md §8 defines these and calls the resulting table a contribution in its own
right. Everything here is numpy/pandas/scipy only -- no jax, no torch, no model imports -- so
that the same functions score Heston today and the neural SDE in Phase 2 without noticing the
difference.

A design rule runs through the whole module: **every Monte-Carlo quantity is reported with
its standard error**, and comparisons are expressed as a z-score against that standard error.
"MC pricing error = 0.42" is uninterpretable on its own -- 0.42 what, against noise of what
size? The z-score is what separates a genuine discretisation bias from sampling noise, and it
is the number to look at first.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from ..algorithms.black_scholes import implied_vol
from ..algorithms.preprocessing.preprocessor import Marginal
from .paths import Paths

logger = getLogger()


# --------------------------------------------------------------------------- #
# Calibration accuracy
# --------------------------------------------------------------------------- #
@dataclass
class SurfaceError:
    """A model's misfit to a cleaned market surface, in both implied-vol and price space.

    Vol space is the one that matters (it is scale-free across strikes and maturities, which
    price errors are not -- a $5 error on a $200 ITM call and on a $2 wing call are not
    comparable), but price RMSE is reported too because it is what the calibration actually
    minimised.

    n_dropped counts quotes whose MODEL price had no valid implied vol, i.e. the model priced
    outside the no-arbitrage bounds. A large count invalidates the RMSE below it.
    """

    n_quotes: int
    n_dropped: int
    iv_rmse: float          # absolute vol units: 0.0123 == 1.23 vol points
    iv_mae: float
    price_rmse: float
    price_rel_mae: float


def surface_rmse(clean: pd.DataFrame, model_call: np.ndarray, r: float) -> SurfaceError:
    """Score model call prices against a cleaned market surface.

    clean      - SurfacePreprocessor.clean() output; supplies forward, strike, ttm and the
                 market `iv` (already inverted there, so only the model side needs work).
    model_call - one model call price per row of `clean`, in the same order.
    """
    model_call = np.asarray(model_call, dtype=float)
    if model_call.shape[0] != len(clean):
        msg = f"model_call has {model_call.shape[0]} prices for {len(clean)} quotes."
        raise ValueError(msg)

    iv_model = implied_vol(model_call, clean["forward"].to_numpy(),
                           clean["strike"].to_numpy(), clean["ttm"].to_numpy(), r)
    iv_market = clean["iv"].to_numpy()
    market_price = clean["call_price"].to_numpy()

    finite = np.isfinite(iv_model) & np.isfinite(iv_market)
    n_dropped = int((~finite).sum())
    if n_dropped:
        logger.warning("surface_rmse: %d/%d quotes priced outside the no-arb bounds by the "
                       "model and had no implied vol.", n_dropped, len(clean))

    iv_err = iv_model[finite] - iv_market[finite]
    price_err = model_call - market_price

    return SurfaceError(
        n_quotes=len(clean),
        n_dropped=n_dropped,
        iv_rmse=float(np.sqrt(np.mean(iv_err ** 2))) if iv_err.size else float("nan"),
        iv_mae=float(np.mean(np.abs(iv_err))) if iv_err.size else float("nan"),
        price_rmse=float(np.sqrt(np.mean(price_err ** 2))),
        price_rel_mae=float(np.mean(np.abs(price_err) / market_price)),
    )


# --------------------------------------------------------------------------- #
# Distributional fidelity
# --------------------------------------------------------------------------- #
@dataclass
class MarginalError:
    """Agreement between a model's simulated S_T and the Breeden-Litzenberger marginal."""

    ttm: float
    w1: float               # 1-Wasserstein distance, in index points
    w1_rel: float           # w1 / E_BL[S_T]: dimensionless, comparable across maturities
    bl_mean: float
    model_mean: float
    model_stderr: float
    tail_mass: float        # fraction of model samples falling OUTSIDE the BL strike grid


def marginal_wasserstein(marginal: Marginal, other: Marginal | np.ndarray) -> float:
    """W1 between a BL marginal and either another density or a raw sample.

    Both cases collapse onto one scipy call. A density on a strike grid IS a discrete measure
    -- values = strikes, weights = density -- and a sample is the same thing with uniform
    weights, so `wasserstein_distance(u_values, v_values, u_weights, v_weights)` handles the
    pair natively and the two grids need not match.

    Density-vs-density is the Phase-1 case (comparing BL marginals to each other); the
    density-vs-samples branch is what Phase 2's neural SDE and Phase 3's scenarios will use.
    """
    if isinstance(other, Marginal):
        return float(wasserstein_distance(marginal.strikes, other.strikes,
                                          marginal.density, other.density))

    samples = np.asarray(other, dtype=float)

    return float(wasserstein_distance(marginal.strikes, samples, marginal.density, None))


def marginal_report(marginals: Sequence[Marginal],
                    samples: Mapping[float, np.ndarray]) -> pd.DataFrame:
    """Per-maturity W1 between the BL marginals and a model's simulated S_T.

    `tail_mass` is the guard-rail: the BL density lives on a finite strike grid, so any model
    sample landing outside it is being compared against a distribution that was truncated and
    renormalised. If tail_mass is not ~0, W1 is measuring the grid, not the model -- widen
    MarginalConfig.strike_hi / strike_lo rather than trusting the number.
    """
    rows = []
    for marginal in marginals:
        drawn = np.asarray(samples[marginal.ttm], dtype=float)
        lo, hi = marginal.strikes[0], marginal.strikes[-1]
        w1 = marginal_wasserstein(marginal, drawn)
        bl_mean = marginal.mean()
        rows.append(MarginalError(
            ttm=marginal.ttm,
            w1=w1,
            w1_rel=w1 / bl_mean,
            bl_mean=bl_mean,
            model_mean=float(drawn.mean()),
            model_stderr=float(drawn.std(ddof=1) / np.sqrt(drawn.size)),
            tail_mass=float(np.mean((drawn < lo) | (drawn > hi))),
        ))

    return pd.DataFrame([vars(row) for row in rows])


# --------------------------------------------------------------------------- #
# Monte Carlo vs the exact pricer
# --------------------------------------------------------------------------- #
def mc_call_prices(paths: Paths, strike: np.ndarray,
                   ttm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Discounted MC call prices and their standard errors on parallel (K, T) arrays.

    Slices the one path bundle at each maturity -- no re-simulation per quote.
    """
    strike = np.asarray(strike, dtype=float)
    ttm = np.asarray(ttm, dtype=float)
    price = np.empty(strike.shape)
    stderr = np.empty(strike.shape)

    for t in np.unique(ttm):
        at_t = ttm == t
        spot = paths.slice_at(float(t))[:, None]                    # (n_paths, 1)
        payoff = np.maximum(spot - strike[at_t][None, :], 0.0) * np.exp(-paths.r * t)
        price[at_t] = payoff.mean(axis=0)
        stderr[at_t] = payoff.std(axis=0, ddof=1) / np.sqrt(payoff.shape[0])

    return price, stderr


@dataclass
class MCPricingError:
    """Agreement between the MC simulator and the exact Fourier pricer.

    z_max is the field to read first: it is max |mc - reference| / stderr, so it says whether
    the gap is Monte-Carlo noise (z of order 1-3) or a real discretisation bias (z >> 3). The
    raw rmse alone cannot distinguish the two.
    """

    n_quotes: int
    rmse: float
    rel_rmse: float
    max_abs_error: float
    mean_stderr: float
    z_max: float


def mc_pricing_error(mc_price: np.ndarray, mc_stderr: np.ndarray,
                     reference_price: np.ndarray) -> MCPricingError:
    """Compare MC prices against an exact reference, in standard-error units."""
    mc_price = np.asarray(mc_price, dtype=float)
    mc_stderr = np.asarray(mc_stderr, dtype=float)
    reference_price = np.asarray(reference_price, dtype=float)

    err = mc_price - reference_price
    # a degenerate bundle (every path identical) has zero spread; a 0/0 z-score is 0, not inf
    safe = np.where(mc_stderr > 0, mc_stderr, np.inf)
    z = np.abs(err) / safe

    return MCPricingError(
        n_quotes=int(err.size),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        rel_rmse=float(np.sqrt(np.mean((err / reference_price) ** 2))),
        max_abs_error=float(np.max(np.abs(err))),
        mean_stderr=float(np.mean(mc_stderr)),
        z_max=float(np.max(z)),
    )


# --------------------------------------------------------------------------- #
# No-arbitrage
# --------------------------------------------------------------------------- #
def martingale_residual(paths: Paths, ttms: Sequence[float]) -> pd.DataFrame:
    """E[S_T] - S_0 e^{rT} per maturity, with its Monte-Carlo standard error.

    The residual alone means nothing. At S_0 ~ 6000, sigma ~ 20%, T = 1 and 50k paths, the
    standard error on E[S_T] is already ~5 index points -- so a residual of 10 points is
    noise, not an arbitrage. `z` is the residual in standard errors and is the column to
    judge on: |z| > 3 is a real martingale violation.

    Note the simulator applies no martingale correction on purpose: rescaling the paths to hit
    the forward would make this metric identically zero and therefore vacuous.
    """
    rows = []
    for ttm in ttms:
        terminal = paths.slice_at(float(ttm))
        mean = float(terminal.mean())
        stderr = float(terminal.std(ddof=1) / np.sqrt(terminal.size))
        forward = paths.s_0 * np.exp(paths.r * ttm)
        residual = mean - forward
        rows.append({
            "ttm": float(ttm),
            "e_s_t": mean,
            "forward": forward,
            "residual": residual,
            "rel_residual": residual / forward,
            "stderr": stderr,
            "z": residual / stderr if stderr > 0 else 0.0,
        })

    return pd.DataFrame(rows)


def bl_martingale_residual(marginals: Sequence[Marginal], s_0: float,
                           r: float) -> pd.DataFrame:
    """Apply the same forward-recovery check to the BL densities themselves.

    This scores the DATA, not a model: a martingale's marginal must have mean E[S_T] = F, so a
    large residual here means the extracted density is wrong before any model touches it.

    Read it together with `raw_mass`, the probability the traded strike range actually captured
    (see Marginal). The two failure modes pull in opposite directions and the pair tells them
    apart: a grid that stops short of the tails gives raw_mass < 1 and a mildly biased mean,
    while a grid running past the last quote reads the density off SVI's extrapolated wing and
    can invent enough fake tail mass to blow the residual out by tens of percent. A residual
    that survives both is genuine -- most likely a single flat r failing to capture the
    per-expiry dividend/rate term structure.
    """
    rows = []
    for marginal in marginals:
        forward = s_0 * np.exp(r * marginal.ttm)
        mean = marginal.mean()
        rows.append({
            "ttm": marginal.ttm,
            "e_s_t": mean,
            "forward": forward,
            "residual": mean - forward,
            "rel_residual": mean / forward - 1.0,
            "raw_mass": marginal.raw_mass,
            "k_lo": float(np.log(marginal.strikes[0] / forward)),
            "k_hi": float(np.log(marginal.strikes[-1] / forward)),
        })

    return pd.DataFrame(rows)
