"""Every figure the notebooks draw.

The notebooks ARE the Phase-1 deliverable, so their figures are the thing most likely to rot
and the thing worth having under test. The numeric work lives in surface.py; this module only
draws, which keeps it thin.

Conventions: every function returns the Figure, accepts an optional `ax`, and never calls
plt.show() or plt.savefig() -- display is the notebook's business, not the library's.

NOT re-exported from src/eval/__init__.py on purpose: importing any submodule of a package
executes its __init__, and heston_mc imports src.eval.paths -- so exporting this would drag
matplotlib onto the import path of every calibration run.
"""
from collections.abc import Sequence
from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..algorithms.preprocessing.preprocessor import Marginal
from ..algorithms.preprocessing.svi import SVIParams
from .paths import Paths
from .surface import iv_surface_grid

logger = getLogger()

SURFACE_3D = "surface"
SURFACE_CONTOUR = "contour"

DENSITY_SPOT = "spot"
DENSITY_VARIANCE = "variance"


def _figure(ax: plt.Axes | None, **kwargs: object) -> tuple[Figure, plt.Axes]:
    """Reuse the caller's axes, or make our own."""
    if ax is not None:
        return ax.figure, ax
    fig, ax = plt.subplots(**kwargs)

    return fig, ax


def plot_iv_surface(clean: pd.DataFrame, svi: list[SVIParams],
                    kind: str = SURFACE_CONTOUR, ax: plt.Axes | None = None) -> Figure:
    """Draw the Phase-1 exit-criterion figure: implied vol over (log-moneyness, maturity).

    The mesh comes from the fitted SVI slices via iv_surface_grid, which interpolates total
    variance (not vol) across maturities so the surface is calendar-arbitrage-free. Market IVs
    are overlaid so the fit can be judged against the quotes rather than admired on its own.

    kind="contour" (default) is the readable version and the one to put in the writeup;
    kind="surface" is the 3D view, which looks impressive and hides detail.
    """
    if kind not in (SURFACE_3D, SURFACE_CONTOUR):
        msg = f"Unknown kind {kind!r}; expected {SURFACE_CONTOUR!r} or {SURFACE_3D!r}."
        raise ValueError(msg)

    k_market = clean["log_moneyness"].to_numpy()
    k_grid, t_grid, iv = iv_surface_grid(svi, k_lo=float(k_market.min()),
                                         k_hi=float(k_market.max()))
    mesh_k, mesh_t = np.meshgrid(k_grid, t_grid)

    if kind == SURFACE_3D:
        fig = plt.figure(figsize=(9, 6)) if ax is None else ax.figure
        ax = fig.add_subplot(111, projection="3d") if ax is None else ax
        ax.plot_surface(mesh_k, mesh_t, iv, cmap="viridis", alpha=0.8,
                        linewidth=0, antialiased=True)
        ax.scatter(k_market, clean["ttm"], clean["iv"], s=4, c="crimson",
                   depthshade=False, label="market")
        ax.set_zlabel("implied vol")
        ax.view_init(elev=22, azim=-135)
    else:
        fig, ax = _figure(ax, figsize=(9, 5.5))
        filled = ax.contourf(mesh_k, mesh_t, iv, levels=24, cmap="viridis")
        ax.contour(mesh_k, mesh_t, iv, levels=12, colors="white", linewidths=0.4, alpha=0.5)
        ax.scatter(k_market, clean["ttm"], s=5, c="crimson", alpha=0.55, label="market quotes")
        fig.colorbar(filled, ax=ax, label="implied vol")
        ax.legend(loc="upper right")

    ax.set_xlabel("log-moneyness  k = log(K / F)")
    ax.set_ylabel("maturity T (years)")
    ax.set_title("SPX implied volatility surface (SVI-fitted)")

    return ax.figure


def plot_smiles(clean: pd.DataFrame, svi: list[SVIParams],
                ttms: Sequence[float] | None = None, ax: plt.Axes | None = None) -> Figure:
    """SVI smiles (lines) against the market implied vols (dots), across the term structure."""
    fig, ax = _figure(ax, figsize=(9, 5))
    chosen = [s for s in svi if ttms is None or any(np.isclose(s.t, ttms))]
    for smile in chosen:
        group = clean[np.isclose(clean["ttm"], smile.t)]
        grid = np.linspace(group["strike"].min(), group["strike"].max(), 200)
        line, = ax.plot(grid, smile.implied_vol(grid), lw=1.5, label=f"T={smile.t:.2f}")
        ax.scatter(group["strike"], group["iv"], s=12, alpha=0.5, color=line.get_color())

    ax.set_xlabel("strike K")
    ax.set_ylabel("implied vol")
    ax.set_title("SVI fit (lines) vs market implied vol (dots)")
    ax.legend(title="maturity")

    return fig


def plot_marginals(marginals: Sequence[Marginal], s_0: float,
                   ax: plt.Axes | None = None) -> Figure:
    """Draw the recovered risk-neutral densities -- the second Phase-1 exit-criterion figure.

    Each curve is labelled with its E[S_T], which should equal the forward: the gap between
    the label and the forward is the forward-recovery error, visible at a glance.
    """
    fig, ax = _figure(ax, figsize=(9, 5))
    for marginal in marginals:
        ax.plot(marginal.strikes, marginal.density, lw=1.5,
                label=f"T={marginal.ttm:.2f}  E[S_T]={marginal.mean():.0f}")
    ax.axvline(s_0, color="black", ls="--", lw=0.8, label=f"spot={s_0:.0f}")

    ax.set_xlabel("strike K")
    ax.set_ylabel("density  $\\mu_T(K)$")
    ax.set_title("Recovered risk-neutral marginals (Breeden-Litzenberger)")
    ax.legend(title="maturity", fontsize=8)

    return fig


def plot_smile_fit(clean: pd.DataFrame, model_iv: np.ndarray, ttm: float,
                   ax: plt.Axes | None = None) -> Figure:
    """One maturity's market smile against a calibrated model's implied vols."""
    fig, ax = _figure(ax, figsize=(8, 5))
    at_t = np.isclose(clean["ttm"], ttm)
    group = clean[at_t]
    order = np.argsort(group["strike"].to_numpy())

    ax.scatter(group["strike"], group["iv"], s=18, alpha=0.7, label="market")
    ax.plot(group["strike"].to_numpy()[order], np.asarray(model_iv)[at_t][order],
            color="crimson", lw=1.5, label="model")

    ax.set_xlabel("strike K")
    ax.set_ylabel("implied vol")
    ax.set_title(f"Calibrated fit at T={ttm:.2f}")
    ax.legend()

    return fig


def plot_posterior(posterior: dict[str, np.ndarray], ax: plt.Axes | None = None) -> Figure:
    """Posterior histograms per parameter.

    Without this the MCMC is invisible in the notebook: a posterior mean printed on its own is
    indistinguishable from a least-squares point estimate.
    """
    names = [n for n in ("v_0", "v_mean", "a", "eta", "rho", "sigma") if n in posterior]
    if ax is not None:
        msg = "plot_posterior draws a grid of axes and cannot draw into a single one."
        raise ValueError(msg)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for axis, name in zip(axes.ravel(), names, strict=False):
        draws = np.asarray(posterior[name]).ravel()
        axis.hist(draws, bins=40, color="steelblue", alpha=0.8)
        axis.axvline(draws.mean(), color="crimson", lw=1.2)
        axis.set_title(name)
    for axis in axes.ravel()[len(names):]:
        axis.set_visible(False)

    fig.suptitle("Heston posterior (NUTS); red line = posterior mean")
    fig.tight_layout()

    return fig


def _empirical_density(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-time-step histogram of the simulated values: row i is p(x, t_i).

    Normalised by the TOTAL path count, not by the count that lands inside `edges` -- which is
    what np.histogram(density=True) would do. The distinction is the same one MarginalConfig
    makes about the Breeden-Litzenberger grid: renormalising to the visible window silently
    inflates every slice that has spread outside it, so a late maturity with a quarter of its
    mass off-frame would be drawn as bright as an early one that is fully contained. Here a row
    integrates to the fraction of paths still in view, so the fading at large t is real
    diffusion leaving the frame rather than an artefact of rescaling.
    """
    n_paths = values.shape[0]
    counts = np.stack([np.histogram(col, bins=edges)[0] for col in values.T])

    return counts / (n_paths * np.diff(edges))


def plot_paths_density(paths: Paths, kind: str = DENSITY_SPOT, n_show: int = 10,
                       n_bins: int = 200, q_clip: float = 0.998, q_vmax: float = 0.99,
                       seed: int = 0, ax: plt.Axes | None = None) -> Figure:
    """Draw simulated trajectories over the density they are sampled from, time on the y-axis.

    The model-agnostic answer to "what does the calibrated process actually look like". Phase 1
    only ever scored functionals of these paths -- prices at T, W1 against the BL marginals,
    E[S_T] -- so the process itself was computed and never seen. This is the picture of it.

    Unlike the textbook OU case there is no analytical p(x, t) to plot underneath: Heston's
    transition density has no closed form, and Phase 2's neural LSV will have none either. So
    the background is the EMPIRICAL density of the same simulation the red paths are drawn
    from, which is the only thing available to a harness that must not know which model it is
    scoring. It is a self-consistency picture, not a validation against truth -- the
    validation is `mc_pricing_error` against the exact Fourier pricer.

    Two scaling choices, both cosmetic but neither obvious:

    - `q_clip` frames the x-axis on a central quantile range of the TERMINAL slice rather than
      on the full min/max. A handful of extreme paths otherwise set the width and squeeze the
      whole distribution into a few pixels.
    - `q_vmax` caps the colour scale. At t=0 every path sits on s_0, so that row is a point
      mass whose density is orders of magnitude above everything later; left uncapped it takes
      the entire colour range and the rest of the image goes black.
    """
    if kind == DENSITY_SPOT:
        values, symbol = paths.spot, "S_t"
    elif kind == DENSITY_VARIANCE:
        if paths.variance is None:
            msg = ("These paths carry no variance (MCConfig.store_variance=False); "
                   "re-simulate with it on to draw kind='variance'.")
            raise ValueError(msg)
        values, symbol = paths.variance, "v_t"
    else:
        msg = f"Unknown kind {kind!r}; expected {DENSITY_SPOT!r} or {DENSITY_VARIANCE!r}."
        raise ValueError(msg)

    tail = (1.0 - q_clip) / 2.0
    lo, hi = np.quantile(values[:, -1], [tail, 1.0 - tail])
    edges = np.linspace(float(lo), float(hi), n_bins + 1)
    density = _empirical_density(values, edges)

    fig, ax = _figure(ax, figsize=(11, 7))
    image = ax.imshow(density, extent=(edges[0], edges[-1], paths.times[-1], paths.times[0]),
                      aspect="auto", cmap="viridis",
                      vmax=float(np.quantile(density[density > 0], q_vmax)))

    rng = np.random.default_rng(seed)
    shown = rng.choice(paths.n_paths, size=min(n_show, paths.n_paths), replace=False)
    for j in shown:
        ax.plot(values[j], paths.times, color="red", alpha=0.6, lw=0.8)

    ax.invert_yaxis()
    ax.set_xlabel(symbol)
    ax.set_ylabel("t  (years)")
    ax.set_title(f"Simulated {symbol} paths over their empirical density "
                 f"({paths.n_paths:,} paths, {len(shown)} shown)")
    fig.colorbar(image, ax=ax, label=f"p({symbol}, t)")

    return fig


def plot_mc_prices(clean: pd.DataFrame, mc_price: np.ndarray, mc_stderr: np.ndarray,
                   reference: np.ndarray, n_sigma: float = 2.0,
                   ax: plt.Axes | None = None) -> Figure:
    """Draw Monte-Carlo call prices against the exact reference, over their standardised gap.

    `mc_pricing_error` reduces this comparison to an RMSE and a worst-case z, which is the
    right thing for the benchmark table but hides WHERE a simulator disagrees. Discretisation
    bias is structured -- it concentrates at the short maturities and in the wings, exactly
    where the QE-vs-Euler choice bites -- and that structure is what separates a scheme error
    from sampling noise.

    Two panels, because on their own the prices cannot show it: across four decades of call
    price the MC dots sit exactly on the reference curve whatever the error is. So the upper
    panel is the prices (what was simulated) and the lower one is z = (mc - ref) / stderr
    (whether it is right). The +/-3 band is the decision line the benchmark's `mc_price_z_max`
    column reports: inside it the gap is Monte-Carlo noise, outside it the scheme is biased.

    Like plot_posterior this draws a panel grid, so it cannot take a caller's single axis.
    """
    if ax is not None:
        msg = "plot_mc_prices draws a price/residual panel pair and cannot draw into one axis."
        raise ValueError(msg)

    fig, (upper, lower) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                       height_ratios=[2, 1])
    strike = clean["strike"].to_numpy()
    mc_price, mc_stderr = np.asarray(mc_price), np.asarray(mc_stderr)
    reference = np.asarray(reference)
    z = (mc_price - reference) / mc_stderr

    for ttm in sorted(clean["ttm"].unique()):
        at_t = np.isclose(clean["ttm"], ttm)
        order = np.argsort(strike[at_t])
        k = strike[at_t][order]

        line, = upper.plot(k, reference[at_t][order], lw=1.2, label=f"T={ttm:.2f}")
        upper.errorbar(k, mc_price[at_t][order], yerr=n_sigma * mc_stderr[at_t][order],
                       fmt="o", ms=3, lw=0.8, capsize=2, color=line.get_color(), alpha=0.7)
        lower.plot(k, z[at_t][order], "o-", ms=3, lw=0.8,
                   color=line.get_color(), alpha=0.8)

    upper.set_yscale("log")
    upper.set_ylabel("call price")
    upper.set_title(f"Monte-Carlo prices (dots, ±{n_sigma:g}·se) vs the exact pricer (lines)")
    upper.legend(title="maturity", fontsize=8)

    lower.axhline(0.0, color="black", lw=0.8)
    for band in (-3.0, 3.0):
        lower.axhline(band, color="crimson", ls="--", lw=0.8)
    lower.set_xlabel("strike K")
    lower.set_ylabel("z = (mc - ref) / se")
    lower.set_title("Standardised residual; outside ±3 is discretisation bias, not noise",
                    fontsize=9)

    fig.tight_layout()

    return fig
