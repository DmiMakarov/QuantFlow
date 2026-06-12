"""Tests for the HestonParams dataclass."""
from src.algorithms.heston.heston_params import HestonParams


def test_fields_are_stored() -> None:
    p = HestonParams(v_0=0.04, v_mean=0.05, a=1.5, eta=0.3, rho=-0.6)
    assert (p.v_0, p.v_mean, p.a, p.eta, p.rho) == (0.04, 0.05, 1.5, 0.3, -0.6)


def test_feller_satisfied() -> None:
    # 2 * a * v_mean = 0.16 >= eta**2 = 0.09
    p = HestonParams(v_0=0.04, v_mean=0.04, a=2.0, eta=0.3, rho=-0.5)
    assert p.feller() is True


def test_feller_violated() -> None:
    # 2 * a * v_mean = 0.04 < eta**2 = 0.25
    p = HestonParams(v_0=0.04, v_mean=0.04, a=0.5, eta=0.5, rho=-0.5)
    assert p.feller() is False
