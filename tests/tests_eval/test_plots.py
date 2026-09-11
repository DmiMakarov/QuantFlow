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
from src.eval.paths import Paths
from src.eval.plots import (
    _empirical_density,
    plot_iv_surface,
    plot_marginals,
    plot_mc_prices,
    plot_paths_density,
    plot_posterior,
    plot_smile_fit,
    plot_smiles,
)

FORWARD = 100.0


def _bar_span(ax) -> float:
    """Total vertical extent of every error bar drawn on the axis."""
    segments = [seg for container in ax.containers
                for seg in container.lines[2][0].get_segments()]

    return float(sum(seg[:, 1].max() - seg[:, 1].min() for seg in segments))


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


@pytest.fixture
def paths() -> Paths:
    """A small GBM-ish bundle: enough spread for the histogram, cheap enough to draw."""
    rng = np.random.default_rng(0)
    times = np.linspace(0.0, 1.0, 13)
    shocks = rng.normal(scale=0.2 * np.sqrt(times[1]), size=(64, times.size - 1))
    spot = FORWARD * np.exp(np.cumsum(np.hstack([np.zeros((64, 1)), shocks]), axis=1))

    variance = 0.04 * np.exp(rng.normal(scale=0.3, size=spot.shape))

    return Paths(times=times, spot=spot, variance=variance, r=0.01)


def test_paths_density_draws_the_image_and_the_shown_trajectories(paths) -> None:
    ax = plot_paths_density(paths, n_show=5).axes[0]

    assert len(ax.images) == 1            # the density heatmap
    assert len(ax.lines) == 5             # one red path each
    assert ax.get_ylabel().startswith("t")
    bottom, top = ax.get_ylim()
    assert bottom < top                   # t increases upward, as in the OU picture
    assert np.isclose(top, paths.times[-1])


def test_paths_density_caps_n_show_at_the_number_of_paths(paths) -> None:
    ax = plot_paths_density(paths, n_show=1000).axes[0]
    assert len(ax.lines) == paths.n_paths


def test_paths_density_can_draw_the_variance(paths) -> None:
    fig = plot_paths_density(paths, kind="variance")
    assert fig.axes[0].get_xlabel() == "v_t"


def test_paths_density_refuses_variance_that_was_not_stored(paths) -> None:
    dropped = Paths(times=paths.times, spot=paths.spot, variance=None, r=paths.r)
    with pytest.raises(ValueError, match="no variance"):
        plot_paths_density(dropped, kind="variance")


def test_paths_density_rejects_an_unknown_kind(paths) -> None:
    with pytest.raises(ValueError, match="Unknown kind"):
        plot_paths_density(paths, kind="volatility")


def test_paths_density_rows_integrate_to_the_fraction_still_in_frame(paths) -> None:
    """Rows must NOT be renormalised to the visible window: a slice that has diffused half its
    mass off-frame has to look half as bright, or the picture hides its own truncation."""
    edges = np.linspace(80.0, 120.0, 41)
    density = _empirical_density(paths.spot, edges)
    inside = np.array([((col >= edges[0]) & (col <= edges[-1])).mean() for col in paths.spot.T])

    assert np.allclose(density.sum(axis=1) * np.diff(edges)[0], inside, atol=1e-12)


def test_mc_prices_draw_prices_over_their_standardised_residual(clean) -> None:
    reference = np.linspace(5.0, 15.0, len(clean))
    mc_price = reference + 0.05
    mc_stderr = np.full(len(clean), 0.02)

    upper, lower = plot_mc_prices(clean, mc_price, mc_stderr, reference).axes

    assert upper.get_yscale() == "log"
    assert "exact pricer" in upper.get_title()
    assert lower.get_ylabel().startswith("z")


def test_mc_prices_residual_panel_shows_the_z_score(clean) -> None:
    """The lower panel is the only part that can reveal a bias, so it must plot mc-vs-ref in
    standard errors -- not the raw gap, which is unreadable across four decades of price."""
    reference = np.full(len(clean), 10.0)
    mc_stderr = np.full(len(clean), 0.25)
    mc_price = reference + 0.5                      # exactly 2 standard errors high

    lower = plot_mc_prices(clean, mc_price, mc_stderr, reference).axes[1]
    drawn = np.concatenate([line.get_ydata() for line in lower.lines
                            if len(line.get_ydata()) == len(clean) // 2])

    assert np.allclose(drawn, 2.0)


def test_mc_prices_error_bars_scale_with_n_sigma(clean) -> None:
    reference = np.full(len(clean), 10.0)
    stderr = np.full(len(clean), 0.5)

    narrow = plot_mc_prices(clean, reference, stderr, reference, n_sigma=1.0).axes[0]
    wide = plot_mc_prices(clean, reference, stderr, reference, n_sigma=3.0).axes[0]

    assert _bar_span(wide) > _bar_span(narrow)


def test_mc_prices_refuses_a_single_axis(clean) -> None:
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="cannot draw into one axis"):
        plot_mc_prices(clean, np.ones(len(clean)), np.ones(len(clean)),
                       np.ones(len(clean)), ax=ax)
