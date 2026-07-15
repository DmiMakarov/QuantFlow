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
from .surface import iv_surface_grid

logger = getLogger()

SURFACE_3D = "surface"
SURFACE_CONTOUR = "contour"


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
