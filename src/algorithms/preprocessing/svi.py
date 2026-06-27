"""Raw-SVI volatility-smile parametrisation and per-maturity fitting.

A single maturity's smile is summarised by Gatheral's raw SVI total-variance form

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) ),   k = log(K / F),

where w = sigma_BS^2 * t is the total implied variance. SVI is smooth and (under mild
parameter constraints) arbitrage-free, so it denoises the market smile and extrapolates
the thin wings -- exactly what the Breeden-Litzenberger second derivative needs to be
stable.
"""
from dataclasses import dataclass
from logging import getLogger

import numpy as np
from scipy.optimize import least_squares

from .preprocessing_config import SVIConfig

logger = getLogger()


@dataclass
class SVIParams:
    """Raw-SVI parameters for one maturity slice.

    a - vertical level of total variance
    b - angle between the left/right wings (>= 0)
    rho - skew / wing asymmetry (|rho| < 1)
    m - horizontal shift of the smile in log-moneyness
    sigma - ATM curvature smoothness (> 0)
    t - maturity (years) this slice was fitted at
    forward - forward price F used to define log-moneyness k = log(K / F)
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    t: float
    forward: float

    def total_variance(self, k: float | np.ndarray) -> float | np.ndarray:
        """Raw-SVI total variance w(k) at log-moneyness k."""
        k = np.asarray(k, dtype=float)

        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, strike: float | np.ndarray) -> float | np.ndarray:
        """Black-Scholes implied vol at the given strike(s)."""
        k = np.log(np.asarray(strike, dtype=float) / self.forward)

        return np.sqrt(np.maximum(self.total_variance(k), 0.0) / self.t)

    def is_arbitrage_free(self) -> bool:
        """Whether the necessary static no-arbitrage conditions hold.

        Checks b >= 0, |rho| < 1, sigma > 0, and that the minimum total variance
        a + b * sigma * sqrt(1 - rho^2) is non-negative. These are necessary (not
        sufficient) -- treat as a diagnostic, like HestonParams.feller().
        """
        min_total_var = self.a + self.b * self.sigma * np.sqrt(1.0 - self.rho ** 2)

        return bool(self.b >= 0.0 and abs(self.rho) < 1.0
                    and self.sigma > 0.0 and min_total_var >= 0.0)


def fit_svi_slice(k: np.ndarray,
                  total_var: np.ndarray,
                  t: float,
                  forward: float,
                  weights: np.ndarray | None = None,
                  config: SVIConfig | None = None) -> SVIParams:
    """Fit raw SVI to one maturity by bounded least squares on total-variance residuals.

    k - log-moneyness of each quote, log(K / F)
    total_var - market total variance w_i = iv_i^2 * t at each quote
    t - maturity (years)
    forward - forward price F
    weights - optional per-quote residual weights (e.g. BS vega); defaults to 1
    config - SVIConfig with bounds / solver settings; defaults reproduce documented values
    """
    cfg = config if config is not None else SVIConfig()
    k = np.asarray(k, dtype=float)
    total_var = np.asarray(total_var, dtype=float)
    w = np.ones_like(total_var) if weights is None else np.asarray(weights, dtype=float)

    lb = np.array(cfg.lb())
    ub = np.array(cfg.ub())
    # equity-skew warm start: flat level at the observed minimum, downward skew.
    x0 = np.clip(np.array([float(total_var.min()), 0.1, -0.5, 0.0, 0.1]), lb, ub)

    def residuals(x: np.ndarray) -> np.ndarray:
        a, b, rho, m, sigma = x
        model = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))

        return w * (model - total_var)

    result = least_squares(residuals, x0, bounds=(lb, ub),
                           method=cfg.method, max_nfev=cfg.max_nfev)
    fitted = SVIParams(*result.x, t=t, forward=forward)
    logger.info("SVI fit: cost=%.3e status=%d arb_free=%s",
                result.cost, result.status, fitted.is_arbitrage_free())

    return fitted
