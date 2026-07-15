"""Tests for the Heston Monte-Carlo simulator.

These are the gate on the whole MC-pricing-error metric: if the simulator is biased, the
metric reports fake model error. So the assertions are anchored to things that are known
EXACTLY -- the Black-Scholes price in the eta -> 0 limit, the Fourier pricer, and the
martingale property -- and every tolerance is expressed in Monte-Carlo standard errors
rather than as a hand-tuned epsilon.
"""
import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.algorithms.black_scholes import bs_call
from src.algorithms.heston.heston_analytics import HestonModel
from src.algorithms.heston.heston_config import (
    HestonCalibrationConfig,
    MCConfig,
    QuadConfig,
)
from src.algorithms.heston.heston_mc import HestonSimulator
from src.algorithms.heston.heston_params import HestonParams

S_0 = 100.0
R = 0.03

# Feller SATISFIED: 2*a*v_mean = 2*2*0.04 = 0.16 >= eta^2 = 0.09
CALM = HestonParams(v_0=0.04, v_mean=0.04, a=2.0, eta=0.3, rho=-0.7)
# Feller VIOLATED: 2*a*v_mean = 0.16 < eta^2 = 1.0 -- the regime QE exists for, and the one
# real calibrated SPX params land in.
ROUGH = HestonParams(v_0=0.04, v_mean=0.04, a=2.0, eta=1.0, rho=-0.7)


def _mc_call(paths, strike: float, ttm: float) -> tuple[float, float]:
    """Discounted MC call price and its standard error."""
    payoff = np.maximum(paths.slice_at(ttm) - strike, 0.0) * np.exp(-paths.r * ttm)

    return float(payoff.mean()), float(payoff.std(ddof=1) / np.sqrt(payoff.size))


# --------------------------------------------------------------------------- #
# Analytic anchors
# --------------------------------------------------------------------------- #
def test_zero_vol_of_vol_degenerates_to_black_scholes(caplog: pytest.LogCaptureFixture) -> None:
    """With eta -> 0 and a = 0 the variance is frozen, so S is exactly lognormal and the
    call has a closed form. Also exercises the eta_min guard: QE is ill-conditioned here and
    must fall back to Euler, which is exact in this limit anyway."""
    sigma = 0.2
    flat = HestonParams(v_0=sigma ** 2, v_mean=sigma ** 2, a=1e-8, eta=1e-9, rho=0.0)
    cfg = MCConfig(n_paths=40_000, steps_per_year=200, seed=1)

    with caplog.at_level(logging.WARNING):
        paths = HestonSimulator(flat, cfg).simulate(S_0, R, 0.5)
    assert "falling back to the Euler scheme" in caplog.text

    ttm, strike = 0.5, 105.0
    price, stderr = _mc_call(paths, strike, ttm)
    exact = float(bs_call(S_0 * np.exp(R * ttm), strike, ttm, sigma, R))

    assert abs(price - exact) < 3 * stderr


@pytest.mark.parametrize("scheme", ["qe", "euler"])
@pytest.mark.parametrize("params", [CALM, ROUGH], ids=["feller_ok", "feller_violated"])
def test_mc_agrees_with_the_fourier_pricer(scheme: str, params: HestonParams) -> None:
    """THE metric, promoted to a gate: the simulator must reproduce the exact Fourier price.

    Tolerance is 4 MC standard errors, so this fails on genuine discretisation bias but not
    on Monte-Carlo noise.
    """
    cfg = MCConfig(scheme=scheme, n_paths=40_000, steps_per_year=200, seed=7)
    paths = HestonSimulator(params, cfg).simulate(S_0, R, [0.25, 0.5])
    pricer = HestonModel(params, HestonCalibrationConfig(quad=QuadConfig(n_quad=128)))

    for ttm in (0.25, 0.5):
        for strike in (90.0, 100.0, 110.0):
            price, stderr = _mc_call(paths, strike, ttm)
            exact = float(pricer.call(np.array([strike]), S_0, R, np.array([ttm]))[0])
            assert abs(price - exact) < 4 * stderr, (
                f"{scheme} K={strike} T={ttm}: mc={price:.4f} exact={exact:.4f} "
                f"z={abs(price - exact) / stderr:.2f}")


@pytest.mark.parametrize("scheme", ["qe", "euler"])
def test_discounted_spot_is_a_martingale(scheme: str) -> None:
    """E[S_T] = S_0 e^{rT}. Checked in standard errors -- at these path counts the SE on the
    mean is a couple of index points, so a raw epsilon would be meaningless."""
    cfg = MCConfig(scheme=scheme, n_paths=40_000, steps_per_year=200, seed=3)
    paths = HestonSimulator(ROUGH, cfg).simulate(S_0, R, 1.0)

    terminal = paths.terminal
    stderr = terminal.std(ddof=1) / np.sqrt(terminal.size)
    forward = S_0 * np.exp(R * 1.0)

    assert abs(terminal.mean() - forward) < 3 * stderr


@pytest.mark.parametrize("scheme", ["qe", "euler"])
def test_feller_violating_params_stay_finite_and_non_negative(scheme: str) -> None:
    """The whole reason QE is the default. Full-truncation Euler must survive here too:
    negative variance may appear inside its increment but must never reach the output."""
    assert not ROUGH.feller()
    cfg = MCConfig(scheme=scheme, n_paths=2_000, steps_per_year=100, seed=5)
    paths = HestonSimulator(ROUGH, cfg).simulate(S_0, R, 1.0)

    assert np.isfinite(paths.spot).all()
    assert np.isfinite(paths.variance).all()
    assert (paths.spot > 0).all()
    assert (paths.variance >= 0).all()


def test_qe_and_euler_converge_to_each_other() -> None:
    """Two independent discretisations of the same SDE must agree as dt -> 0. This catches a
    scheme-specific bug (e.g. miswiring QE's rho) that a single-scheme test cannot."""
    common = {"n_paths": 40_000, "steps_per_year": 400, "seed": 11}
    qe = HestonSimulator(CALM, MCConfig(scheme="qe", **common)).simulate(S_0, R, 0.5)
    euler = HestonSimulator(CALM, MCConfig(scheme="euler", **common)).simulate(S_0, R, 0.5)

    p_qe, se_qe = _mc_call(qe, 100.0, 0.5)
    p_eu, se_eu = _mc_call(euler, 100.0, 0.5)

    assert abs(p_qe - p_eu) < 4 * np.hypot(se_qe, se_eu)


# --------------------------------------------------------------------------- #
# Mechanics
# --------------------------------------------------------------------------- #
def test_every_requested_maturity_lands_exactly_on_the_grid() -> None:
    """A maturity that misses the grid cannot be sliced, so the union in _time_grid is
    load-bearing -- 0.37 is not a multiple of 1/252."""
    ttms = [0.37, 0.5, 1.0]
    paths = HestonSimulator(CALM, MCConfig(n_paths=100)).simulate(S_0, R, ttms)

    for ttm in ttms:
        assert paths.slice_at(ttm).shape == (100,)
    assert paths.times[0] == 0.0
    assert np.all(np.diff(paths.times) > 0)          # strictly increasing


def test_same_seed_reproduces_paths_and_different_seed_does_not() -> None:
    def run(seed: int) -> np.ndarray:
        return HestonSimulator(CALM, MCConfig(n_paths=200, seed=seed)).simulate(
            S_0, R, 0.5).spot

    np.testing.assert_array_equal(run(0), run(0))
    assert not np.array_equal(run(0), run(1))


def test_antithetic_rounds_an_odd_path_count_up(caplog: pytest.LogCaptureFixture) -> None:
    cfg = MCConfig(antithetic=True, n_paths=101)
    with caplog.at_level(logging.INFO):
        paths = HestonSimulator(CALM, cfg).simulate(S_0, R, 0.25)

    assert paths.n_paths == 102
    assert "rounding up" in caplog.text


def test_antithetic_can_be_switched_off() -> None:
    cfg = MCConfig(antithetic=False, n_paths=101)
    assert HestonSimulator(CALM, cfg).simulate(S_0, R, 0.25).n_paths == 101


def test_variance_can_be_dropped() -> None:
    cfg = MCConfig(store_variance=False, n_paths=100)
    assert HestonSimulator(CALM, cfg).simulate(S_0, R, 0.25).variance is None


def test_unknown_scheme_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="Unknown scheme"):
        HestonSimulator(CALM, MCConfig(scheme="milstein"))


def test_non_positive_maturity_raises() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        HestonSimulator(CALM, MCConfig(n_paths=100)).simulate(S_0, R, [0.0, 0.5])


def test_gamma_2_is_derived_from_gamma_1() -> None:
    assert MCConfig(gamma_1=0.5).gamma_2 == 0.5
    assert MCConfig(gamma_1=1.0).gamma_2 == 0.0


# --------------------------------------------------------------------------- #
# Property-based (PROJECT_PLAN.md §6 asks for these on the SDE samplers)
# --------------------------------------------------------------------------- #
@settings(max_examples=15, deadline=None)
@given(
    v_0=st.floats(min_value=0.005, max_value=0.5),
    v_mean=st.floats(min_value=0.005, max_value=0.5),
    a=st.floats(min_value=0.1, max_value=8.0),
    eta=st.floats(min_value=0.05, max_value=3.0),
    rho=st.floats(min_value=-0.95, max_value=0.95),
    scheme=st.sampled_from(["qe", "euler"]),
)
def test_paths_stay_valid_for_any_admissible_params(
    v_0: float, v_mean: float, a: float, eta: float, rho: float, scheme: str,
) -> None:
    """Across the whole admissible parameter box -- Feller-violating corners included -- the
    simulator must never produce a nan, an inf, a non-positive spot, or a negative variance."""
    params = HestonParams(v_0=v_0, v_mean=v_mean, a=a, eta=eta, rho=rho)
    cfg = MCConfig(scheme=scheme, n_paths=200, steps_per_year=50, seed=0)
    paths = HestonSimulator(params, cfg).simulate(S_0, R, 1.0)

    assert np.isfinite(paths.spot).all()
    assert (paths.spot > 0).all()
    assert np.isfinite(paths.variance).all()
    assert (paths.variance >= 0).all()
