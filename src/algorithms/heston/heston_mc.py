"""Monte-Carlo path simulator for the risk-neutral Heston dynamics.

Companion to HestonModel's exact Fourier pricer (heston_analytics.py): the pricer gives the
reference price, this gives the paths. Phase 1 needs it for the MC-pricing-error and
martingale-residual metrics -- both of which are validated *against* that pricer -- and
Phase 3 reuses it unchanged as the deep-hedging scenario generator, which is why it returns
whole paths rather than just terminal values.

Two schemes:

  QE (Andersen 2008, default) -- samples the variance from a moment-matched quadratic /
  exponential proxy rather than discretising its SDE, so it stays accurate when the Feller
  condition 2*a*v_mean >= eta^2 fails. Calibrated SPX params routinely violate Feller, and
  that is precisely where Euler acquires a visible bias. Since the MC-vs-Fourier gap is a
  reported metric, a biased scheme would masquerade as model error.

  Euler full-truncation (Lord et al. 2010) -- the naive scheme, kept as a cross-check. QE
  and Euler must agree as dt -> 0, and both must agree with the Fourier pricer.

numpy rather than jax/torch on purpose: no gradient is ever taken through the simulator
(deep hedging backprops through the *policy*, on pre-simulated paths), so a jax dependency
would buy nothing and cost a handoff at the Phase-3 boundary.
"""
from collections.abc import Sequence
from logging import getLogger

import numpy as np

from ...eval.paths import Paths
from .heston_config import SCHEME_EULER, SCHEME_QE, MCConfig
from .heston_params import HestonParams

logger = getLogger()

_TINY = 1e-300   # floor for the moment-matching denominator; only bites if v_mean == 0


class HestonSimulator:
    """Simulates Heston paths under the risk-neutral measure, emitting `Paths`."""

    def __init__(self, params: HestonParams, config: MCConfig | None = None) -> None:
        """Store params and config. Unknown schemes fail here, not mid-simulation."""
        self.params = params
        self.config = config if config is not None else MCConfig()
        if self.config.scheme not in (SCHEME_QE, SCHEME_EULER):
            msg = (f"Unknown scheme {self.config.scheme!r}; "
                   f"expected {SCHEME_QE!r} or {SCHEME_EULER!r}.")
            raise ValueError(msg)

    # ------------------------------------------------------------------- scheme selection
    def _scheme(self) -> str:
        """Resolve the scheme actually used, downgrading QE to Euler at tiny vol-of-vol.

        QE's log-spot coefficients carry rho/eta and a*v_mean/eta terms which cancel
        analytically as eta -> 0 but catastrophically in floating point. Euler is exact in
        that limit anyway (the variance stops moving), so falling back loses nothing.
        """
        if self.config.scheme == SCHEME_QE and self.params.eta < self.config.eta_min:
            logger.warning("eta=%.2e below eta_min=%.2e; QE is ill-conditioned there, "
                           "falling back to the Euler scheme.",
                           self.params.eta, self.config.eta_min)
            return SCHEME_EULER

        return self.config.scheme

    # ------------------------------------------------------------------- grid and draws
    @staticmethod
    def _time_grid(maturities: np.ndarray, steps_per_year: int) -> np.ndarray:
        """Uniform grid to the longest maturity, unioned with the maturities themselves.

        Every requested maturity must land EXACTLY on a grid point, or Paths.index_of will
        (rightly) refuse to slice it. The union makes dt non-uniform at the seams, which both
        schemes handle since they take dt per step.
        """
        horizon = float(maturities.max())
        n_uniform = max(int(np.ceil(horizon * steps_per_year)), 1)
        uniform = np.linspace(0.0, horizon, n_uniform + 1)
        grid = np.union1d(uniform, maturities)

        return np.unique(np.concatenate([[0.0], grid]))

    def _draws(self, rng: np.random.Generator, n_paths: int,
               n_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(z_v, z_x, u) blocks, antithetically paired when configured.

        The antithetic twin negates both normals and reflects the uniform (u -> 1 - u). Under
        QE the branch taken depends on psi, which diverges between twins after the first step,
        so the pairing is only formally antithetic -- but each half is still marginally correct,
        so the estimator stays UNBIASED; only the variance reduction is not guaranteed.
        """
        if not self.config.antithetic:
            return (rng.standard_normal((n_paths, n_steps)),
                    rng.standard_normal((n_paths, n_steps)),
                    rng.random((n_paths, n_steps)))

        half = n_paths // 2
        z_v = rng.standard_normal((half, n_steps))
        z_x = rng.standard_normal((half, n_steps))
        u = rng.random((half, n_steps))

        return (np.vstack([z_v, -z_v]), np.vstack([z_x, -z_x]), np.vstack([u, 1.0 - u]))

    # ------------------------------------------------------------------- the two schemes
    def _euler_step(self, v: np.ndarray, x: np.ndarray, dt: float, r: float,
                    z_v: np.ndarray, z_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One full-truncation Euler step on (v, log S).

        Full truncation (Lord et al.): the variance is floored at 0 wherever it is *used* but
        allowed to go negative in its own increment. Truncating the state instead reintroduces
        a positive bias in the variance.

        Here -- unlike QE -- the spot's Brownian IS correlated with the variance's: the rho
        coupling enters through z_x = rho*z_v + sqrt(1 - rho^2)*z_perp.
        """
        p = self.params
        v_pos = np.maximum(v, 0.0)
        sqrt_v_dt = np.sqrt(v_pos * dt)

        z_s = p.rho * z_v + np.sqrt(1.0 - p.rho ** 2) * z_x
        v_next = v + p.a * (p.v_mean - v_pos) * dt + p.eta * sqrt_v_dt * z_v
        x_next = x + (r - 0.5 * v_pos) * dt + sqrt_v_dt * z_s

        return v_next, x_next

    def _qe_step(self, v: np.ndarray, x: np.ndarray, dt: float, r: float,
                 z_v: np.ndarray, z_x: np.ndarray,
                 u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One Andersen QE step on (v, log S).

        The variance is sampled from a proxy distribution moment-matched to the exact
        (non-central chi-squared) transition: a quadratic-in-a-normal when the coefficient of
        variation psi is small, an exponential-with-an-atom-at-zero when it is large. Both
        branches are evaluated for every path and selected with np.where -- clamping psi
        BEFORE each branch is what keeps the unselected one from emitting nan/inf warnings.

        The log-spot uses Andersen's central discretisation. Note z_x is INDEPENDENT of z_v:
        the rho correlation is carried by the K1*v + K2*v_next terms, not by a correlated
        normal. Wiring a correlated normal in here is the classic QE bug.
        """
        p, cfg = self.params, self.config
        exp_at = np.exp(-p.a * dt)

        # moments of v_{t+dt} | v_t under the exact transition law
        m = p.v_mean + (v - p.v_mean) * exp_at
        s2 = (v * p.eta ** 2 * exp_at / p.a) * (1.0 - exp_at) \
            + (p.v_mean * p.eta ** 2 / (2.0 * p.a)) * (1.0 - exp_at) ** 2
        m = np.maximum(m, _TINY)
        psi = s2 / (m * m)

        # quadratic branch (psi <= psi_c): v = a_q * (b_q + z)^2
        psi_lo = np.minimum(psi, cfg.psi_c)
        inv = 2.0 / psi_lo
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a_q = m / (1.0 + b2)
        v_quad = a_q * (np.sqrt(b2) + z_v) ** 2

        # exponential branch (psi > psi_c): inverse-CDF of an atom at 0 plus an exponential
        psi_hi = np.maximum(psi, cfg.psi_c)
        prob = (psi_hi - 1.0) / (psi_hi + 1.0)
        beta = (1.0 - prob) / m
        v_exp = np.where(u <= prob, 0.0, np.log((1.0 - prob) / (1.0 - u)) / beta)

        v_next = np.maximum(np.where(psi <= cfg.psi_c, v_quad, v_exp), 0.0)

        # Andersen's central log-spot discretisation; r*dt is the risk-neutral drift on top
        # of the martingale-case coefficients.
        g_1, g_2 = cfg.gamma_1, cfg.gamma_2
        k_0 = -p.rho * p.a * p.v_mean * dt / p.eta
        k_1 = g_1 * dt * (p.a * p.rho / p.eta - 0.5) - p.rho / p.eta
        k_2 = g_2 * dt * (p.a * p.rho / p.eta - 0.5) + p.rho / p.eta
        k_3 = g_1 * dt * (1.0 - p.rho ** 2)
        k_4 = g_2 * dt * (1.0 - p.rho ** 2)

        x_next = (x + r * dt + k_0 + k_1 * v + k_2 * v_next
                  + np.sqrt(np.maximum(k_3 * v + k_4 * v_next, 0.0)) * z_x)

        return v_next, x_next

    # ------------------------------------------------------------------- orchestration
    def simulate(self, s_0: float, r: float, maturities: float | Sequence[float],
                 n_paths: int | None = None, seed: int | None = None) -> Paths:
        """Simulate to max(maturities) on a grid that lands exactly on every maturity.

        One bundle serves the whole surface -- slice it per maturity rather than re-simulating
        per quote. Deliberately does NOT apply a martingale correction: the martingale residual
        E[S_T] - S_0*e^{rT} is a *reported* metric, and rescaling the paths to hit the forward
        would drive it to zero by construction and make it vacuous.
        """
        cfg = self.config
        n_paths = cfg.n_paths if n_paths is None else n_paths
        seed = cfg.seed if seed is None else seed

        if cfg.antithetic and n_paths % 2:
            n_paths += 1
            logger.info("Antithetic sampling needs an even path count; rounding up to %d.",
                        n_paths)

        ttms = np.atleast_1d(np.asarray(maturities, dtype=float))
        if np.any(ttms <= 0):
            msg = "All maturities must be strictly positive."
            raise ValueError(msg)

        times = self._time_grid(ttms, cfg.steps_per_year)
        n_steps = times.size - 1
        rng = np.random.default_rng(seed)
        z_v, z_x, u = self._draws(rng, n_paths, n_steps)
        scheme = self._scheme()

        spot = np.empty((n_paths, n_steps + 1))
        variance = np.empty((n_paths, n_steps + 1))
        x = np.full(n_paths, np.log(s_0))
        v = np.full(n_paths, self.params.v_0)
        spot[:, 0], variance[:, 0] = s_0, v

        for i in range(n_steps):
            dt = float(times[i + 1] - times[i])
            if scheme == SCHEME_QE:
                v, x = self._qe_step(v, x, dt, r, z_v[:, i], z_x[:, i], u[:, i])
            else:
                v, x = self._euler_step(v, x, dt, r, z_v[:, i], z_x[:, i])
            # Full-truncation Euler deliberately lets v go negative inside its own increment
            # (truncating the state would bias the variance upward), but a negative variance
            # must never escape into the Paths contract -- floor what we publish, not what we
            # recurse on. Under QE v is already non-negative, so this is a no-op there.
            spot[:, i + 1], variance[:, i + 1] = np.exp(x), np.maximum(v, 0.0)

        return Paths(times=times, spot=spot,
                     variance=variance if cfg.store_variance else None, r=r)
