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


@dataclass
class MarginalConfig:
    """Breeden-Litzenberger extraction grid and smoothing settings."""

    n_grid: int = 400              # number of strikes on the dense BL grid
    strike_lo: float = 0.2         # grid lower bound as a fraction of the forward
    strike_hi: float = 3.0         # grid upper bound as a fraction of the forward
    # wide enough that the lognormal-ish tails carry negligible truncated mass, so the
    # extracted density integrates to ~1 and recovers the forward as its mean.
    kernel_bandwidth: float = 0.0  # Gaussian smoothing sigma (in grid steps); 0 = off
    normalize: bool = True         # rescale density to integrate to 1


@dataclass
class PreprocessConfig:
    """Top-level config bundling the liquidity, arbitrage, SVI, and marginal settings."""

    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    svi: SVIConfig = field(default_factory=SVIConfig)
    marginal: MarginalConfig = field(default_factory=MarginalConfig)
