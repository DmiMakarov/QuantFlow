"""Tests for the Phase-1 metrics.

Anchored on exact analytic properties wherever one exists -- a Wasserstein distance between
two densities shifted by delta must BE delta, a zero-error surface must score exactly zero --
rather than on tolerances tuned until they passed.
"""
import numpy as np
import pandas as pd
import pytest

from src.algorithms.black_scholes import bs_call
from src.algorithms.preprocessing.preprocessor import Marginal
from src.eval.metrics import (
    bl_martingale_residual,
    marginal_report,
    marginal_wasserstein,
    martingale_residual,
    mc_call_prices,
    mc_pricing_error,
    surface_rmse,
)
from src.eval.paths import Paths

FORWARD = 100.0
R = 0.0


def _clean(iv: float = 0.2) -> pd.DataFrame:
    """A cleaned surface whose call_price is CONSISTENT with its iv, so a model that
    reproduces the prices scores exactly zero."""
    strike = np.array([90.0, 100.0, 110.0])
    ttm = np.full(3, 1.0)
    return pd.DataFrame({
        "strike": strike,
        "ttm": ttm,
        "forward": FORWARD,
        "log_moneyness": np.log(strike / FORWARD),
        "iv": iv,
        "call_price": bs_call(FORWARD, strike, ttm, iv, R),
    })


def _gaussian_marginal(mean: float, sd: float = 10.0, ttm: float = 1.0,
                       n: int = 4001) -> Marginal:
    grid = np.linspace(mean - 8 * sd, mean + 8 * sd, n)
    density = np.exp(-0.5 * ((grid - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    return Marginal(ttm=ttm, strikes=grid, density=density)


# --------------------------------------------------------------------------- #
# surface_rmse
# --------------------------------------------------------------------------- #
def test_perfect_model_scores_exactly_zero() -> None:
    clean = _clean()
    err = surface_rmse(clean, clean["call_price"].to_numpy(), R)

    assert err.iv_rmse == pytest.approx(0.0, abs=1e-9)
    assert err.price_rmse == pytest.approx(0.0, abs=1e-12)
    assert err.n_dropped == 0
    assert err.n_quotes == 3


def test_a_known_vol_offset_comes_back_as_that_offset() -> None:
    """Price the surface at 25% against a 20% market: iv_rmse must be exactly 5 vol points."""
    clean = _clean(iv=0.20)
    model = bs_call(FORWARD, clean["strike"].to_numpy(), clean["ttm"].to_numpy(), 0.25, R)
    err = surface_rmse(clean, model, R)

    assert err.iv_rmse == pytest.approx(0.05, abs=1e-6)
    assert err.iv_mae == pytest.approx(0.05, abs=1e-6)


def test_model_prices_outside_no_arb_bounds_are_dropped_and_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model price above the forward has no implied vol; it must be excluded from the vol
    RMSE and COUNTED, because a large n_dropped invalidates the number above it."""
    clean = _clean()
    model = clean["call_price"].to_numpy().copy()
    model[0] = 10 * FORWARD                                  # unattainable
    with caplog.at_level("WARNING"):
        err = surface_rmse(clean, model, R)

    assert err.n_dropped == 1
    assert "outside the no-arb bounds" in caplog.text
    assert np.isfinite(err.iv_rmse)                          # scored on the survivors


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="prices for"):
        surface_rmse(_clean(), np.array([1.0, 2.0]), R)


# --------------------------------------------------------------------------- #
# marginal_wasserstein  (the metric that did not exist anywhere in the repo)
# --------------------------------------------------------------------------- #
def test_w1_between_two_shifted_gaussians_is_the_shift() -> None:
    """W1 has a closed form for a pure translation: it IS the shift. No tolerance-fudging."""
    left, right = _gaussian_marginal(100.0), _gaussian_marginal(103.0)
    assert marginal_wasserstein(left, right) == pytest.approx(3.0, abs=1e-3)


def test_w1_of_a_density_with_itself_is_zero() -> None:
    marginal = _gaussian_marginal(100.0)
    assert marginal_wasserstein(marginal, marginal) == pytest.approx(0.0, abs=1e-12)


def test_w1_works_against_raw_samples_not_just_densities() -> None:
    """The density-vs-samples branch is the one Phase 2's neural SDE will hit."""
    marginal = _gaussian_marginal(100.0, sd=10.0)
    samples = np.random.default_rng(0).normal(103.0, 10.0, 200_000)

    assert marginal_wasserstein(marginal, samples) == pytest.approx(3.0, abs=0.1)


def test_w1_is_insensitive_to_the_grids_matching() -> None:
    """The two measures need not share a strike grid -- W1 is defined between measures, and
    scipy's implementation takes values + weights precisely so this works."""
    coarse = _gaussian_marginal(100.0, n=201)
    fine = _gaussian_marginal(103.0, n=4001)

    assert marginal_wasserstein(coarse, fine) == pytest.approx(3.0, abs=1e-2)


def test_marginal_report_flags_samples_falling_off_the_bl_grid() -> None:
    """tail_mass is the guard-rail: if the model's samples land outside the BL support, W1 is
    scoring the truncation of the grid rather than the model."""
    narrow = Marginal(ttm=1.0, strikes=np.linspace(95.0, 105.0, 101),
                      density=np.ones(101) / 10.0)
    samples = {1.0: np.array([50.0, 100.0, 100.0, 150.0])}   # half the mass is off-grid
    report = marginal_report([narrow], samples)

    assert report.loc[0, "tail_mass"] == pytest.approx(0.5)
    assert report.loc[0, "ttm"] == 1.0
    assert report.loc[0, "w1"] > 0


def test_marginal_report_columns() -> None:
    marginal = _gaussian_marginal(100.0)
    samples = {1.0: np.random.default_rng(1).normal(100.0, 10.0, 5_000)}
    report = marginal_report([marginal], samples)

    assert list(report.columns) == ["ttm", "w1", "w1_rel", "bl_mean", "model_mean",
                                    "model_stderr", "tail_mass"]
    assert report.loc[0, "w1_rel"] == pytest.approx(report.loc[0, "w1"]
                                                    / report.loc[0, "bl_mean"])


# --------------------------------------------------------------------------- #
# MC pricing error
# --------------------------------------------------------------------------- #
def _flat_paths(terminal: np.ndarray, ttm: float = 1.0, r: float = 0.0) -> Paths:
    spot = np.column_stack([np.full(terminal.size, 100.0), terminal])

    return Paths(times=np.array([0.0, ttm]), spot=spot, r=r)


def test_mc_call_prices_are_the_discounted_mean_payoff() -> None:
    paths = _flat_paths(np.array([90.0, 110.0, 130.0]), ttm=1.0, r=0.10)
    price, stderr = mc_call_prices(paths, np.array([100.0]), np.array([1.0]))

    payoffs = np.array([0.0, 10.0, 30.0]) * np.exp(-0.10)
    assert price[0] == pytest.approx(payoffs.mean())
    assert stderr[0] == pytest.approx(payoffs.std(ddof=1) / np.sqrt(3))


def test_mc_pricing_error_reports_the_gap_in_standard_errors() -> None:
    """z_max is the whole point: an rmse of 0.4 is meaningless until you know the noise."""
    err = mc_pricing_error(mc_price=np.array([10.4, 20.0]),
                           mc_stderr=np.array([0.1, 0.1]),
                           reference_price=np.array([10.0, 20.0]))

    assert err.n_quotes == 2
    assert err.max_abs_error == pytest.approx(0.4)
    assert err.z_max == pytest.approx(4.0)                    # 0.4 / 0.1 -> real bias
    assert err.rmse == pytest.approx(np.sqrt((0.4 ** 2 + 0.0) / 2))
    assert err.mean_stderr == pytest.approx(0.1)


def test_zero_stderr_yields_a_zero_z_not_an_infinity() -> None:
    """A degenerate bundle (every path identical) has no spread; 0/0 must be 0, not inf."""
    err = mc_pricing_error(np.array([10.0]), np.array([0.0]), np.array([10.0]))
    assert err.z_max == 0.0


# --------------------------------------------------------------------------- #
# Martingale residual
# --------------------------------------------------------------------------- #
def test_martingale_residual_is_reported_with_its_standard_error() -> None:
    rng = np.random.default_rng(0)
    terminal = 100.0 * np.exp(rng.normal(-0.5 * 0.04, 0.2, 20_000))   # E[S_T] = 100, r = 0
    paths = _flat_paths(terminal, ttm=1.0, r=0.0)
    table = martingale_residual(paths, [1.0])

    assert list(table.columns) == ["ttm", "e_s_t", "forward", "residual",
                                   "rel_residual", "stderr", "z"]
    assert table.loc[0, "forward"] == pytest.approx(100.0)
    assert abs(table.loc[0, "z"]) < 3.0            # a true martingale: noise, not violation


def test_martingale_residual_z_is_zero_for_a_degenerate_bundle() -> None:
    paths = _flat_paths(np.full(5, 100.0), ttm=1.0, r=0.0)
    assert martingale_residual(paths, [1.0]).loc[0, "z"] == 0.0


def test_bl_martingale_residual_scores_the_data_not_a_model() -> None:
    """A BL density whose mean IS the forward has zero residual."""
    marginal = _gaussian_marginal(100.0)
    table = bl_martingale_residual([marginal], s_0=100.0, r=0.0)

    assert table.loc[0, "residual"] == pytest.approx(0.0, abs=1e-6)
    assert table.loc[0, "rel_residual"] == pytest.approx(0.0, abs=1e-8)
    assert list(table.columns) == ["ttm", "e_s_t", "forward", "residual",
                                   "rel_residual", "raw_mass", "k_lo", "k_hi"]


def test_bl_martingale_residual_surfaces_the_truncated_mass() -> None:
    """raw_mass is what distinguishes a truncated grid from a genuinely bad density, so it has
    to be carried through to the table rather than recomputed (post-normalisation it is always
    1.0 and therefore says nothing)."""
    truncated = Marginal(ttm=1.0, strikes=np.linspace(90.0, 110.0, 101),
                         density=np.ones(101) / 20.0, raw_mass=0.87)
    table = bl_martingale_residual([truncated], s_0=100.0, r=0.0)

    assert table.loc[0, "raw_mass"] == 0.87
    assert table.loc[0, "k_lo"] < 0 < table.loc[0, "k_hi"]
