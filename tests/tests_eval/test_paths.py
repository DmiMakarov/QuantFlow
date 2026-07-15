"""Tests for the Paths scenario contract."""
import numpy as np
import pytest

from src.eval.paths import Paths


@pytest.fixture
def paths() -> Paths:
    times = np.array([0.0, 0.25, 0.5])
    spot = np.array([[100.0, 101.0, 102.0],
                     [100.0, 99.0, 98.0]])
    variance = np.full((2, 3), 0.04)

    return Paths(times=times, spot=spot, variance=variance, r=0.05)


def test_scalar_properties(paths: Paths) -> None:
    assert paths.s_0 == 100.0
    assert paths.n_paths == 2
    assert paths.terminal.tolist() == [102.0, 98.0]


def test_slice_at_returns_the_column_for_that_maturity(paths: Paths) -> None:
    assert paths.slice_at(0.25).tolist() == [101.0, 99.0]
    assert paths.index_of(0.5) == 2


def test_off_grid_maturity_raises_rather_than_snapping(paths: Paths) -> None:
    """Snapping to the nearest column would price a DIFFERENT maturity than asked for, and
    the mistake would surface as an unexplained bias in the pricing metrics, not a crash."""
    with pytest.raises(ValueError, match="not on the simulated grid"):
        paths.index_of(0.3)


def test_maturity_matched_within_float_tolerance(paths: Paths) -> None:
    """The grid is built by arithmetic, so an exact == against a caller's float is fragile."""
    assert paths.index_of(0.25 + 1e-12) == 1


def test_samples_is_the_dict_metrics_consume(paths: Paths) -> None:
    out = paths.samples([0.25, 0.5])

    assert list(out) == [0.25, 0.5]
    assert out[0.5].tolist() == [102.0, 98.0]


def test_variance_is_optional() -> None:
    bare = Paths(times=np.array([0.0, 1.0]), spot=np.ones((3, 2)))
    assert bare.variance is None
    assert bare.r == 0.0
