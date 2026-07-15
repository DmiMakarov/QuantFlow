"""Hyperparameter configs for option-surface preprocessing; defaults are sensible v1 values.

Mirrors the heston_config.py pattern: small dataclasses bundled into one top-level
PreprocessConfig via default_factory.
"""
from dataclasses import dataclass, field

# raw-SVI parameter order, mirrors SVIParams field order in svi.py
SVI_PARAM_ORDER: tuple[str, ...] = ("a", "b", "rho", "m", "sigma")


@dataclass
class LiquidityConfig:
    """Thresholds for dropping untradeable quotes before fitting."""

    min_volume: int = 1            # require at least this traded volume
    min_open_interest: int = 1     # require at least this open interest
    max_rel_spread: float = 0.5    # drop quotes with (ask - bid) / mid above this
    drop_zero_bid: bool = True     # a zero/absent bid means no real market


@dataclass
class ArbitrageConfig:
    """Tolerances for the static no-arbitrage filters."""

    butterfly_tol: float = 1e-8    # allowed slope decrease before flagging non-convexity
    calendar_tol: float = 1e-8     # allowed total-variance decrease across maturities


@dataclass
class SVIConfig:
    """Bounds and scipy.least_squares settings for the per-maturity raw-SVI fit.

    Bounds are (low, high) per param in SVI_PARAM_ORDER (a, b, rho, m, sigma).
    """

    bounds: tuple[tuple[float, float], ...] = (
        (-1.0, 5.0),       # a   (vertical level of total variance)
        (1e-4, 10.0),      # b   (>= 0: angle between the wings)
        (-0.999, 0.999),   # rho (skew)
        (-2.0, 2.0),       # m   (horizontal shift, in log-moneyness)
        (1e-4, 5.0),       # sigma (>0: smoothness of the ATM curvature)
    )
    method: str = "trf"
    max_nfev: int = 10000
    vega_weight: bool = True       # weight residuals by BS vega (fit ~ in IV space)

    def lb(self) -> list[float]:
        """Lower bounds in SVI_PARAM_ORDER."""
        return [lo for lo, _ in self.bounds]

    def ub(self) -> list[float]:
        """Upper bounds in SVI_PARAM_ORDER."""
        return [hi for _, hi in self.bounds]


GRID_TRADED = "traded"
GRID_FIXED = "fixed"


@dataclass
class MarginalConfig:
    """Breeden-Litzenberger extraction grid and smoothing settings.

    The grid width is the single most consequential choice here, and it is not obvious.

    BL reads the density off the SVI smile's curvature, so wherever the grid runs past the last
    traded strike the density is being read off *extrapolation*, not data. SVI's wings grow
    linearly in total variance, and on real SPX slices the fitted right wing is steep enough
    that extending the grid invents tail mass out of nothing. Because the mean weights by K,
    that fake mass dominates: on the June-2026 chain, a grid running to 1.5F or 3F blows the
    forward-recovery error out to +23-27% at the mid maturities, while a grid stopping at the
    last traded strike holds it under 1% everywhere.

    So the default is GRID_TRADED: per maturity, span exactly the strikes that actually traded
    (optionally widened by margin_sigma ATM standard deviations). The extracted density is then
    the risk-neutral density *conditional on the traded range*, renormalised -- and `raw_mass`
    on each Marginal reports how much probability the range failed to capture, so the
    truncation is stated rather than hidden.

    GRID_FIXED restores the old fraction-of-forward grid via strike_lo/strike_hi. It is kept
    for synthetic surfaces (where the smile IS the true model and extrapolating is safe), not
    because it is a reasonable choice on market data.
    """

    n_grid: int = 400              # number of strikes on the dense BL grid
    grid_mode: str = GRID_TRADED
    margin_sigma: float = 0.0      # traded mode: widen by margin_sigma * atm_iv * sqrt(t)
    strike_lo: float = 0.2         # fixed mode only: bounds as a fraction of the forward
    strike_hi: float = 3.0
    kernel_bandwidth: float = 0.0  # Gaussian smoothing sigma (in grid steps); 0 = off
    normalize: bool = True         # rescale density to integrate to 1


@dataclass
class PreprocessConfig:
    """Top-level config bundling the liquidity, arbitrage, SVI, and marginal settings."""

    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    svi: SVIConfig = field(default_factory=SVIConfig)
    marginal: MarginalConfig = field(default_factory=MarginalConfig)
