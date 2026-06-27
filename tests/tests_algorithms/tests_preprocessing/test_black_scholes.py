"""Tests for the forward-parametrised Black-Scholes helpers."""
import numpy as np

from src.algorithms.black_scholes import (
    bs_call,
    bs_vega,
    implied_vol,
    implied_vol_scalar,
)


def test_bs_call_matches_known_atm_value() -> None:
    # F = K = 100, t = 1, sigma = 0.2, r = 0 -> 100 * (N(0.1) - N(-0.1)) ~= 7.9656
    price = bs_call(100.0, 100.0, 1.0, 0.2, 0.0)
    assert price == np.float64(price)
    assert abs(float(price) - 7.9656) < 1e-2


def test_bs_call_discounts_with_rate() -> None:
    # a positive rate raises the forward-form call vs r = 0 at the same spot-forward.
    no_rate = float(bs_call(100.0, 100.0, 1.0, 0.2, 0.0))
    with_rate = float(bs_call(100.0 * np.exp(0.05), 100.0, 1.0, 0.2, 0.05))
    assert with_rate > no_rate


def test_bs_vega_positive_and_peaks_near_atm() -> None:
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    vega = bs_vega(100.0, strikes, 1.0, 0.2, 0.0)
    assert np.all(vega > 0)
    assert vega.argmax() == 2  # ATM strike carries the most vega


def test_implied_vol_round_trip() -> None:
    price = bs_call(100.0, 110.0, 0.75, 0.3, 0.01)
    recovered = implied_vol_scalar(float(price), 100.0, 110.0, 0.75, 0.01)
    assert abs(recovered - 0.3) < 1e-6


def test_implied_vol_below_intrinsic_is_nan() -> None:
    # F = 100, K = 90, r = 0 -> intrinsic = 10; a price of 5 cannot be inverted.
    assert np.isnan(implied_vol_scalar(5.0, 100.0, 90.0, 1.0, 0.0))


def test_implied_vol_above_upper_bound_is_nan() -> None:
    # price >= e^{-rt} F = 100 has no implied vol.
    assert np.isnan(implied_vol_scalar(101.0, 100.0, 100.0, 1.0, 0.0))


def test_implied_vol_brentq_failure_is_nan() -> None:
    # 99.9 is inside (0, 100) but above the sigma=5 call value, so brentq brackets
    # two same-signed endpoints and raises -> the except branch returns nan.
    assert np.isnan(implied_vol_scalar(99.9, 100.0, 100.0, 1.0, 0.0))


def test_implied_vol_vectorised_shape_and_values() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    prices = bs_call(100.0, strikes, 1.0, 0.25, 0.0)
    iv = implied_vol(prices, 100.0, strikes, 1.0, 0.0)
    assert iv.shape == strikes.shape
    assert np.allclose(iv, 0.25, atol=1e-6)
