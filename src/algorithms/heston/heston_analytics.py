"""Analytical (semi-closed-form) pricing of European options under Heston.

Baseline reference: *An Analysis of the Heston Stochastic Volatility Model:
Implementation and Calibration using Matlab*, Ricardo Crisostomo (2014).

The Heston risk-neutral dynamics are

    dS_t = r S_t dt + sqrt(v_t) S_t dW_1
    dv_t = kappa (theta - v_t) dt + sigma sqrt(v_t) dW_2
    d<W_1, W_2>_t = rho dt

A European call does NOT require simulating these SDEs. Heston (1993) showed the
price has a semi-closed form: the characteristic function of the log terminal
price ln(S_T) is known analytically, and the two exercise probabilities Pi_1,
Pi_2 are recovered from it by Fourier inversion. The price is then

    C(K, T) = S_0 * Pi_1 - K * exp(-r * tau) * Pi_2

Here S_0 (today's spot) is an INPUT; S_T is the random variable whose risk-neutral
law the characteristic function encodes. We never "solve for S" -- the output is
the option value.

Numerical note: this module uses the Albrecher et al. ("Little Heston Trap")
formulation of C_j, D_j, which keeps the complex logarithm on its principal
branch and so stays stable for long maturities, unlike Heston's original 1993
sign convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Risk-neutral Heston parameters.

    Attributes:
        v0:    initial variance v_0 (= sigma_0^2), > 0.
        kappa: mean-reversion speed of the variance, > 0.
        theta: long-run variance, > 0.
        sigma: vol-of-vol (volatility of the variance process), > 0.
        rho:   correlation between the two Brownian motions, in [-1, 1].
    """

    v0: float
    kappa: float
    theta: float
    sigma: float
    rho: float

    def feller(self) -> bool:
        """Feller condition 2*kappa*theta >= sigma^2 (variance stays > 0)."""
        return 2.0 * self.kappa * self.theta >= self.sigma**2


def _cf_components(
    phi: np.ndarray,
    j: int,
    p: HestonParams,
    r: float,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (C_j, D_j) of the Heston characteristic function exponent.

    f_j(phi) = exp(C_j + D_j * v0 + i * phi * ln(S0)).  The j in {1, 2} selects
    the two probabilities; they differ only through u_j and b_j. Uses the Little
    Trap (negative-d) formulation. The market price of volatility risk lambda is
    folded into the risk-neutral parameters (lambda = 0).
    """
    i = 1j
    u = 0.5 if j == 1 else -0.5
    b = p.kappa - p.rho * p.sigma if j == 1 else p.kappa
    a = p.kappa * p.theta
    rsi = p.rho * p.sigma * phi * i

    d = np.sqrt((rsi - b) ** 2 - p.sigma**2 * (2.0 * u * phi * i - phi**2))
    # g = c_j of the Little Trap: use (b - rsi - d)/(b - rsi + d).
    g = (b - rsi - d) / (b - rsi + d)

    exp_dt = np.exp(-d * tau)
    C = r * phi * i * tau + (a / p.sigma**2) * (
        (b - rsi - d) * tau - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
    )
    D = ((b - rsi - d) / p.sigma**2) * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
    return C, D


def _char_func(
    phi: np.ndarray,
    j: int,
    p: HestonParams,
    s0: float,
    r: float,
    tau: float,
) -> np.ndarray:
    """Heston characteristic function f_j(phi) of ln(S_T)."""
    C, D = _cf_components(phi, j, p, r, tau)
    return np.exp(C + D * p.v0 + 1j * phi * np.log(s0))


def _probability(
    j: int,
    p: HestonParams,
    s0: float,
    k: float,
    r: float,
    tau: float,
    *,
    n_nodes: int = 128,
    upper: float = 200.0,
) -> float:
    """Exercise probability Pi_j via Gauss-Legendre quadrature.

        Pi_j = 1/2 + (1/pi) * integral_0^inf Re[ e^{-i phi ln K} f_j(phi) / (i phi) ] dphi

    The integral is truncated at `upper` and evaluated with `n_nodes` Legendre
    nodes (which lie strictly inside (0, upper), so the 1/(i phi) singularity at
    phi = 0 is never hit). Defaults (200, 128) match ~1e-8 absolute accuracy for
    typical equity-index parameters.
    """
    # Legendre nodes/weights on [-1, 1], affinely mapped to [0, upper].
    x, w = np.polynomial.legendre.leggauss(n_nodes)
    phi = 0.5 * upper * (x + 1.0)
    weights = 0.5 * upper * w

    f = _char_func(phi, j, p, s0, r, tau)
    integrand = np.real(np.exp(-1j * phi * np.log(k)) * f / (1j * phi))
    return 0.5 + (1.0 / np.pi) * np.dot(weights, integrand)


def heston_call_price(
    p: HestonParams,
    s0: float,
    k: float,
    r: float,
    tau: float,
    *,
    n_nodes: int = 128,
    upper: float = 200.0,
) -> float:
    """European call price under Heston.

    Args:
        p:     Heston parameters.
        s0:    spot price S_0 (input, observed today).
        k:     strike K.
        r:     continuously-compounded risk-free rate.
        tau:   time to maturity in years.
        n_nodes, upper: Fourier-quadrature controls (see `_probability`).

    Returns:
        Call value C(K, tau) = S_0 * Pi_1 - K * exp(-r tau) * Pi_2.
    """
    pi1 = _probability(1, p, s0, k, r, tau, n_nodes=n_nodes, upper=upper)
    pi2 = _probability(2, p, s0, k, r, tau, n_nodes=n_nodes, upper=upper)
    return s0 * pi1 - k * np.exp(-r * tau) * pi2


def heston_put_price(
    p: HestonParams,
    s0: float,
    k: float,
    r: float,
    tau: float,
    *,
    n_nodes: int = 128,
    upper: float = 200.0,
) -> float:
    """European put price via put-call parity: P = C - S_0 + K e^{-r tau}."""
    call = heston_call_price(p, s0, k, r, tau, n_nodes=n_nodes, upper=upper)
    return call - s0 + k * np.exp(-r * tau)
