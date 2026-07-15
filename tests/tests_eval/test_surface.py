"""Tests for the wing split, the IV mesh, and SVI residuals."""
import numpy as np
import pandas as pd
import pytest

from src.algorithms.preprocessing.svi import SVIParams
from src.eval.eval_config import SplitConfig
from src.eval.surface import iv_surface_grid, split_wings, svi_iv_rmse, svi_residuals

FORWARD = 100.0


def _slice(ttm: float, k: np.ndarray) -> pd.DataFrame:
    """One maturity's worth of cleaned quotes at the given log-moneyness points."""
    return pd.DataFrame({
        "ttm": ttm,
        "log_moneyness": k,
        "strike": FORWARD * np.exp(k),
        "forward": FORWARD,
        "iv": 0.2 + 0.1 * k ** 2,
        "call_price": 5.0,
    })


def _flat_svi(ttm: float, total_var: float) -> SVIParams:
    """A degenerate SVI slice with constant total variance `total_var` (b = 0)."""
    return SVIParams(a=total_var, b=0.0, rho=0.0, m=0.0, sigma=0.1, t=ttm, forward=FORWARD)


# --------------------------------------------------------------------------- #
# split_wings
# --------------------------------------------------------------------------- #
def test_quantile_split_holds_out_the_extreme_moneyness_quotes() -> None:
    clean = _slice(1.0, np.linspace(-0.4, 0.4, 9))
    split = split_wings(clean, SplitConfig(wing_frac=0.25, min_train=2))

    # the wings are the |k|-extremes on BOTH sides, not one tail
    assert split.test["log_moneyness"].abs().min() > split.train["log_moneyness"].abs().max()
    assert len(split.train) + len(split.test) == len(clean)


def test_split_is_per_maturity_so_every_slice_contributes_to_both_sets() -> None:
    """A global split would hand the model whole maturities it never saw -- that measures
    term-structure extrapolation, which is a different question from wing fit."""
    clean = pd.concat([_slice(0.5, np.linspace(-0.3, 0.3, 9)),
                       _slice(2.0, np.linspace(-0.3, 0.3, 9))], ignore_index=True)
    split = split_wings(clean, SplitConfig(wing_frac=0.25, min_train=2))

    assert set(split.train["ttm"]) == {0.5, 2.0}
    assert set(split.test["ttm"]) == {0.5, 2.0}


def test_band_mode_holds_out_by_absolute_moneyness() -> None:
    clean = _slice(1.0, np.array([-0.30, -0.10, 0.0, 0.05, 0.20]))
    split = split_wings(clean, SplitConfig(mode="band", k_lo=-0.15, k_hi=0.10, min_train=2))

    assert sorted(split.test["log_moneyness"]) == [-0.30, 0.20]
    assert sorted(split.train["log_moneyness"]) == [-0.10, 0.0, 0.05]


def test_slice_too_thin_to_split_is_kept_whole_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Holding out a wing must never leave a maturity with too few quotes to calibrate on."""
    clean = _slice(1.0, np.linspace(-0.2, 0.2, 4))
    with caplog.at_level("WARNING"):
        split = split_wings(clean, SplitConfig(wing_frac=0.25, min_train=5))

    assert len(split.test) == 0
    assert len(split.train) == 4
    assert "too thin to hold out a wing" in caplog.text


def test_unknown_split_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown split mode"):
        split_wings(_slice(1.0, np.linspace(-0.2, 0.2, 9)), SplitConfig(mode="random"))


def test_split_defaults_are_usable_without_a_config() -> None:
    split = split_wings(_slice(1.0, np.linspace(-0.4, 0.4, 20)))
    assert len(split.test) > 0
    assert len(split.train) > 0


# --------------------------------------------------------------------------- #
# iv_surface_grid
# --------------------------------------------------------------------------- #
def test_mesh_recovers_a_flat_vol_surface_exactly() -> None:
    """Constant sigma across maturities means w = sigma^2 * t is LINEAR in t, which linear
    interpolation of w reproduces exactly. Any leakage shows up immediately."""
    sigma = 0.25
    svi = [_flat_svi(t, sigma ** 2 * t) for t in (0.5, 1.0, 2.0)]
    _, _, iv = iv_surface_grid(svi, n_k=5, n_t=7)

    assert iv == pytest.approx(sigma, abs=1e-12)


def test_mesh_interpolates_total_variance_not_vol() -> None:
    """Interpolating IV directly across maturities can make total variance DECREASE with t,
    which is calendar arbitrage -- the very thing the preprocessor filters quotes for. Build a
    case where the two choices disagree and assert we took the arbitrage-free one."""
    svi = [_flat_svi(1.0, 0.04), _flat_svi(2.0, 0.05)]      # w: 0.04 -> 0.05, iv: 0.20 -> 0.158
    _, t_grid, iv = iv_surface_grid(svi, n_k=3, n_t=3)

    mid = np.searchsorted(t_grid, 1.5)
    w_mid = iv[mid, 0] ** 2 * t_grid[mid]
    assert w_mid == pytest.approx(0.045, abs=1e-12)         # linear in w
    assert iv[mid, 0] != pytest.approx(0.179, abs=1e-3)     # NOT linear in iv

    # total variance must be non-decreasing along t: no calendar arbitrage in the mesh
    total_var = iv ** 2 * t_grid[:, None]
    assert np.all(np.diff(total_var, axis=0) >= -1e-12)


def test_mesh_shapes_and_bounds() -> None:
    svi = [_flat_svi(t, 0.04 * t) for t in (0.5, 1.0)]
    k_grid, t_grid, iv = iv_surface_grid(svi, n_k=11, n_t=6, k_lo=-0.2, k_hi=0.1)

    assert iv.shape == (6, 11)
    assert (k_grid[0], k_grid[-1]) == (-0.2, 0.1)
    assert (t_grid[0], t_grid[-1]) == (0.5, 1.0)


def test_empty_svi_list_raises() -> None:
    with pytest.raises(ValueError, match="No SVI slices"):
        iv_surface_grid([])


# --------------------------------------------------------------------------- #
# svi_residuals
# --------------------------------------------------------------------------- #
def test_residuals_are_zero_when_the_market_lies_exactly_on_the_smile() -> None:
    k = np.linspace(-0.2, 0.2, 7)
    smile = _flat_svi(1.0, 0.04)                            # constant iv = 0.2
    clean = _slice(1.0, k)
    clean["iv"] = 0.2                                       # market == model

    resid = svi_residuals(clean, [smile])
    assert resid["resid"].to_numpy() == pytest.approx(0.0, abs=1e-12)
    assert svi_iv_rmse(clean, [smile]) == pytest.approx(0.0, abs=1e-12)


def test_rmse_is_the_root_mean_square_of_a_known_offset() -> None:
    k = np.linspace(-0.2, 0.2, 5)
    clean = _slice(1.0, k)
    clean["iv"] = 0.2 - 0.01                                # market sits 1 vol point below
    smile = _flat_svi(1.0, 0.04)                            # model at 0.2 flat

    assert svi_iv_rmse(clean, [smile]) == pytest.approx(0.01, abs=1e-12)
    assert list(svi_residuals(clean, [smile]).columns) == [
        "ttm", "strike", "log_moneyness", "iv_market", "iv_svi", "resid"]


def test_rmse_is_nan_when_no_quote_has_a_finite_residual() -> None:
    clean = _slice(1.0, np.linspace(-0.2, 0.2, 5))
    clean["iv"] = np.nan

    assert np.isnan(svi_iv_rmse(clean, [_flat_svi(1.0, 0.04)]))
