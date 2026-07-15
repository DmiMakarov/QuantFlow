"""Tests for the notebook figures.

The notebooks are the Phase-1 deliverable, so a figure that silently stops drawing is a
deliverable regression. These assert structure (a figure came back, the axes are labelled, the
market data actually got drawn), not pixels.
"""
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")   # headless: must precede any pyplot import
import matplotlib.pyplot as plt

from src.algorithms.preprocessing.preprocessor import Marginal
from src.algorithms.preprocessing.svi import SVIParams
from src.eval.plots import (
    plot_iv_surface,
    plot_marginals,
    plot_posterior,
    plot_smile_fit,
    plot_smiles,
)

FORWARD = 100.0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")    # else matplotlib warns after 20 open figures


@pytest.fixture
def clean() -> pd.DataFrame:
    k = np.linspace(-0.2, 0.15, 8)
    return pd.concat([
        pd.DataFrame({"ttm": t, "log_moneyness": k, "strike": FORWARD * np.exp(k),
                      "forward": FORWARD, "iv": 0.2 + 0.1 * k ** 2, "call_price": 5.0})
        for t in (0.5, 1.0)
    ], ignore_index=True)


@pytest.fixture
def svi() -> list[SVIParams]:
    return [SVIParams(a=0.04 * t, b=0.02, rho=-0.4, m=0.0, sigma=0.1, t=t, forward=FORWARD)
            for t in (0.5, 1.0)]


def test_iv_surface_contour_draws_the_mesh_and_the_market(clean, svi) -> None:
    fig = plot_iv_surface(clean, svi, kind="contour")
    ax = fig.axes[0]

    assert "implied volatility surface" in ax.get_title()
    assert ax.get_xlabel().startswith("log-moneyness")
    assert len(ax.collections) > 0        # the contour + the market scatter


def test_iv_surface_3d_variant_draws(clean, svi) -> None:
    fig = plot_iv_surface(clean, svi, kind="surface")
    assert fig.axes[0].get_zlabel() == "implied vol"


def test_iv_surface_rejects_an_unknown_kind(clean, svi) -> None:
    with pytest.raises(ValueError, match="Unknown kind"):
        plot_iv_surface(clean, svi, kind="heatmap")


def test_smiles_draw_one_line_per_maturity(clean, svi) -> None:
    ax = plot_smiles(clean, svi).axes[0]
    assert len(ax.lines) == 2
    assert len(ax.collections) == 2       # market dots per maturity


def test_smiles_can_be_restricted_to_chosen_maturities(clean, svi) -> None:
    ax = plot_smiles(clean, svi, ttms=[1.0]).axes[0]
    assert len(ax.lines) == 1


def test_marginals_draw_a_curve_per_maturity_plus_the_spot_line() -> None:
    grid = np.linspace(50.0, 150.0, 200)
    density = np.exp(-0.5 * ((grid - 100.0) / 10.0) ** 2)
    marginals = [Marginal(ttm=t, strikes=grid, density=density) for t in (0.5, 1.0)]

    ax = plot_marginals(marginals, s_0=100.0).axes[0]
    assert len(ax.lines) == 3             # 2 densities + the spot axvline
    assert "Breeden-Litzenberger" in ax.get_title()


def test_smile_fit_overlays_model_on_market(clean) -> None:
    model_iv = np.full(len(clean), 0.21)
    ax = plot_smile_fit(clean, model_iv, ttm=1.0).axes[0]

    assert len(ax.lines) == 1             # the model curve
    assert len(ax.collections) == 1       # the market scatter
    assert "T=1.00" in ax.get_title()


def test_posterior_draws_a_panel_per_parameter() -> None:
    rng = np.random.default_rng(0)
    posterior = {name: rng.normal(size=200)
                 for name in ("v_0", "v_mean", "a", "eta", "rho", "sigma")}
    fig = plot_posterior(posterior)

    assert sum(ax.get_visible() for ax in fig.axes) == 6


def test_posterior_hides_unused_panels_when_fewer_params() -> None:
    posterior = {"v_0": np.zeros(10), "rho": np.zeros(10)}
    fig = plot_posterior(posterior)

    assert sum(ax.get_visible() for ax in fig.axes) == 2


def test_posterior_refuses_a_single_axis() -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="grid of axes"):
        plot_posterior({"v_0": np.zeros(10)}, ax=ax)


def test_plots_draw_into_a_caller_supplied_axis(clean, svi) -> None:
    """Composing several panels into one figure is the notebook's job, so every plot must
    accept an `ax` rather than always minting its own figure."""
    fig, (left, right) = plt.subplots(1, 2)
    plot_smiles(clean, svi, ax=left)
    plot_iv_surface(clean, svi, kind="contour", ax=right)

    assert left.figure is fig
    assert right.figure is fig
    assert len(left.lines) == 2
