"""Forward-parametrised Black-Scholes helpers shared across the library.

Everything is written in terms of the forward F = s_0 * exp(r * t) (dividends are
folded into F), so an SVI total-variance smile plugs straight in: sigma = sqrt(w / t).

These mirror the private helpers inside HestonModel (heston_analytics.py); they live
here so the preprocessing surface tools can reuse them without importing the Heston
pricer (which pulls in jax/numpyro). Heston keeps its own copies for now -- de-dup is
a future cleanup, intentionally out of scope here.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(forward: np.ndarray, strike: np.ndarray, t: np.ndarray,
           sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Black-Scholes d1, d2 in forward form."""
    sqrt_t = np.sqrt(t)
    d1 = (np.log(forward / strike) + 0.5 * sigma * sigma * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    return d1, d2


def bs_call(forward: float | np.ndarray,
            strike: float | np.ndarray,
            t: float | np.ndarray,
            sigma: float | np.ndarray,
            r: float) -> float | np.ndarray:
    """Black-Scholes call price: e^{-rt} [F N(d1) - K N(d2)]."""
    d1, d2 = _d1_d2(np.asarray(forward, dtype=float), np.asarray(strike, dtype=float),
                    np.asarray(t, dtype=float), np.asarray(sigma, dtype=float))

    return np.exp(-r * np.asarray(t)) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def bs_vega(forward: float | np.ndarray,
            strike: float | np.ndarray,
            t: float | np.ndarray,
            sigma: float | np.ndarray,
            r: float) -> float | np.ndarray:
    """Black-Scholes vega (sensitivity to sigma); used to weight fits into ~IV units."""
    d1, _ = _d1_d2(np.asarray(forward, dtype=float), np.asarray(strike, dtype=float),
                   np.asarray(t, dtype=float), np.asarray(sigma, dtype=float))

    return np.exp(-r * np.asarray(t)) * forward * norm.pdf(d1) * np.sqrt(t)


def implied_vol_scalar(price: float,
                       forward: float,
                       strike: float,
                       t: float,
                       r: float) -> float:
    """Invert one call price to its Black-Scholes implied vol.

    Returns nan when the price violates the no-arbitrage bounds -- it must lie strictly
    between the discounted intrinsic value e^{-rt} max(F - K, 0) and the upper bound
    e^{-rt} F -- since no real implied vol exists there.
    """
    disc = np.exp(-r * t)
    intrinsic = disc * max(forward - strike, 0.0)
    upper = disc * forward
    if price <= intrinsic or price >= upper:
        return np.nan
    try:
        return brentq(lambda sig: bs_call(forward, strike, t, sig, r) - price,
                      1e-6, 5.0, maxiter=100)
    except (ValueError, RuntimeError):
        return np.nan


def implied_vol(price: np.ndarray,
                forward: np.ndarray,
                strike: np.ndarray,
                t: np.ndarray,
                r: float) -> np.ndarray:
    """Vectorised Black-Scholes implied vol over parallel quote arrays."""
    price, forward, strike, t = np.broadcast_arrays(
        np.asarray(price, dtype=float), np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float), np.asarray(t, dtype=float))
    iv = [implied_vol_scalar(p, f, k, tt, r)
          for p, f, k, tt in zip(price.ravel(), forward.ravel(),
                                 strike.ravel(), t.ravel(), strict=True)]

    return np.asarray(iv, dtype=float).reshape(price.shape)
