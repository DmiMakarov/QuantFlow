"""Hyperparameter configs for Heston calibration; defaults reproduce legacy behavior."""
from dataclasses import dataclass, field

PARAM_ORDER: tuple[str, ...] = ("v_0", "v_mean", "a", "eta", "rho")  # mirrors _pack


@dataclass
class QuadConfig:
    """Gauss-Legendre quadrature grid for the characteristic-function inversion.

    u_max is the truncation of the inversion integral, and it is NOT safe to set by feel. The
    Heston characteristic function decays roughly like exp(-c * v * t * u^2), so the shorter the
    maturity the slower it dies in u and the further out the integral has to run. The original
    (u_max=200, n_quad=128) is fine at a year but badly unconverged at the short end: on the
    calibrated June-2026 SPX params it carries a 0.63 ABSOLUTE price error at T=0.017, versus
    ~1e-8 at T=1. That error feeds straight into the calibration objective and into any
    MC-vs-Fourier comparison, where it masquerades as model error.

    (u_max=800, n_quad=256) holds the error below ~1e-5 across the whole 0.017-1.0 maturity
    ladder. It costs 2x the quadrature nodes; correctness is worth more than the constant.
    """

    u_max: float = 800.0
    n_quad: int = 256


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


SCHEME_QE = "qe"
SCHEME_EULER = "euler"


@dataclass
class MCConfig:
    """Settings for the Heston Monte-Carlo path simulator (heston_mc.HestonSimulator).

    scheme defaults to Andersen (2008) QE rather than Euler because calibrated SPX params
    routinely violate the Feller condition, and that is exactly the regime where Euler
    full-truncation picks up a visible discretisation bias. Since the MC-vs-Fourier gap is
    itself a reported metric, a biased scheme would show up as fake model error. "euler" is
    kept as a cross-check: the two must agree as dt -> 0.
    """

    scheme: str = SCHEME_QE
    n_paths: int = 50_000
    steps_per_year: int = 252
    antithetic: bool = True       # pair each draw with its negation to cut E[S_T] noise
    psi_c: float = 1.5            # QE branch-switching threshold (Andersen's recommendation)
    gamma_1: float = 0.5          # QE log-spot weight; gamma_2 = 1 - gamma_1 (central scheme)
    eta_min: float = 1e-3         # below this vol-of-vol QE's 1/eta terms lose precision
    seed: int = 0
    store_variance: bool = True   # retain v_t paths (Phase-3 hedging features need them)

    @property
    def gamma_2(self) -> float:
        """Companion weight to gamma_1. Derived, not configurable: they must sum to 1."""
        return 1.0 - self.gamma_1


@dataclass
class HestonCalibrationConfig:
    """Top-level config bundling the quadrature, MSE, and NUTS settings."""

    quad: QuadConfig = field(default_factory=QuadConfig)
    mse: MSEConfig = field(default_factory=MSEConfig)
    nuts: NUTSConfig = field(default_factory=NUTSConfig)
