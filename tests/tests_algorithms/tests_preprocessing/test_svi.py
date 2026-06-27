"""Tests for the raw-SVI parametrisation and per-slice fit."""
import numpy as np

from src.algorithms.preprocessing.svi import SVIParams, fit_svi_slice


def _true_slice() -> SVIParams:
    return SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.1, t=1.0, forward=100.0)


def test_total_variance_matches_formula() -> None:
    sp = _true_slice()
    k = 0.2
    expected = 0.04 + 0.1 * (-0.3 * (k - 0.0) + np.sqrt((k - 0.0) ** 2 + 0.1 ** 2))
    assert abs(float(sp.total_variance(k)) - expected) < 1e-12


def test_implied_vol_consistent_with_total_variance() -> None:
    sp = _true_slice()
    strike = 110.0
    k = np.log(strike / sp.forward)
    expected = np.sqrt(sp.total_variance(k) / sp.t)
    assert abs(float(sp.implied_vol(strike)) - float(expected)) < 1e-12


def test_is_arbitrage_free_true() -> None:
    assert _true_slice().is_arbitrage_free() is True


def test_is_arbitrage_free_false_on_negative_min_variance() -> None:
    sp = SVIParams(a=-0.5, b=0.01, rho=0.0, m=0.0, sigma=0.01, t=1.0, forward=100.0)
    assert sp.is_arbitrage_free() is False


def test_fit_recovers_noise_free_slice() -> None:
    true = _true_slice()
    k = np.linspace(-0.3, 0.3, 15)
    w = true.total_variance(k)
    fitted = fit_svi_slice(k, w, t=true.t, forward=true.forward)
    # the fitted curve should reproduce the (noise-free) total variance closely.
    assert np.max(np.abs(fitted.total_variance(k) - w)) < 1e-3
    assert fitted.t == true.t
    assert fitted.forward == true.forward


def test_fit_accepts_explicit_weights() -> None:
    true = _true_slice()
    k = np.linspace(-0.2, 0.2, 11)
    w = true.total_variance(k)
    weights = np.linspace(1.0, 2.0, k.size)
    fitted = fit_svi_slice(k, w, t=true.t, forward=true.forward, weights=weights)
    assert np.max(np.abs(fitted.total_variance(k) - w)) < 1e-3
