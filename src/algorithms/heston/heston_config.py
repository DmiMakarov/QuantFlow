"""Hyperparameter configs for Heston calibration; defaults reproduce legacy behavior."""
from dataclasses import dataclass, field

PARAM_ORDER: tuple[str, ...] = ("v_0", "v_mean", "a", "eta", "rho")  # mirrors _pack


@dataclass
class QuadConfig:
    """Gauss-Legendre quadrature grid for the characteristic-function inversion."""

    u_max: float = 200.0          # was HestonModel._U_MAX
    n_quad: int = 128             # was HestonModel._N_QUAD


@dataclass
class MSEConfig:
    """Bounds and scipy.least_squares settings for the MSE (point-estimate) stage."""

    bounds: tuple[tuple[float, float], ...] = (
        (1e-4, 1.0),      # v_0
        (1e-4, 1.0),      # v_mean
        (1e-2, 20.0),     # a
        (1e-2, 5.0),      # eta
        (-0.999, 0.999),  # rho
    )
    method: str = "trf"
    x_scale: str = "jac"
    ftol: float = 1e-8
    xtol: float = 1e-8
    nan_penalty: float = 1.0

    def lb(self) -> list[float]:
        """Lower bounds in PARAM_ORDER."""
        return [lo for lo, _ in self.bounds]

    def ub(self) -> list[float]:
        """Upper bounds in PARAM_ORDER."""
        return [hi for _, hi in self.bounds]


@dataclass
class NUTSConfig:
    """Sampler settings and prior hyperparameters for the Bayesian (NUTS) stage."""

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4                    # was hardcoded in MCMC(...)
    seed: int = 0
    lognormal_scale: float = 0.5           # LogNormal scale for v_0/v_mean/a/eta priors
    rho_low: float = -0.999                # dist.Uniform low (matches MSE rho bound)
    rho_high: float = 0.999
    sigma_halfnormal_scale: float = 0.05   # dist.HalfNormal(...) obs-noise prior
    init_sigma: float = 0.02               # init_to_value sigma seed
    vega_floor: float = 1e-4               # np.maximum(vega, ...)


@dataclass
class HestonCalibrationConfig:
    """Top-level config bundling the quadrature, MSE, and NUTS settings."""

    quad: QuadConfig = field(default_factory=QuadConfig)
    mse: MSEConfig = field(default_factory=MSEConfig)
    nuts: NUTSConfig = field(default_factory=NUTSConfig)
