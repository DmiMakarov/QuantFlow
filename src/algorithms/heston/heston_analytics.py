"""Implementation of analytical solution of Heston Model.

Baseline in An Analysis of the Heston Stochastic Volatility Model:
Implementation and Calibration using Matlab* Ricardo Crisóstomo.
"""
from logging import getLogger

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import brentq, least_squares
from scipy.stats import norm

from .heston_config import HestonCalibrationConfig
from .heston_params import HestonParams

jax.config.update("jax_enable_x64", True)

logger = getLogger()

class HestonModel:
    """Heston model implementation with analytical solution."""

    def __init__(self,
                 init_params: HestonParams,
                 config: HestonCalibrationConfig | None = None) -> None:
        """Define initial Heston parameters.

        v_0 - initial variance
        v_mean - long-term variance
        a - the variance mean-reversion speed
        eta - the volatility of the variance process
        rho - correlation coefficient of Weiner processes

        config bundles the quadrature/MSE/NUTS hyperparameters; defaults reproduce
        the legacy behavior.
        """
        if(not init_params.feller()):
            msg: str = "Bad value of the params: Feller condition isnt satisfied."
            logger.warning(msg)

        self.params = init_params
        self.config = config if config is not None else HestonCalibrationConfig()

    @staticmethod
    def _pack(params: HestonParams) -> jnp.ndarray:
        """Pack HestonParams into the [v_0, v_mean, a, eta, rho] vector the JAX core expects."""
        return jnp.asarray([params.v_0, params.v_mean, params.a, params.eta, params.rho])

    def characteristic_function(self,
                                s_0: float,
                                r: float,
                                t: float | np.ndarray,
                                w: float | np.ndarray,
                                params: HestonParams | None = None) -> complex | np.ndarray:
        """Evaluate the Heston characteristic function (numpy-facing wrapper over `_cf_jax`).

        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        params = params if params is not None else self.params
        cf = self._cf_jax(self._pack(params), s_0, r, jnp.asarray(t), jnp.asarray(w))

        return np.asarray(cf)

    def call(self,
             k: float | np.ndarray,
             s_0: float,
             r: float,
             t: float | np.ndarray,
             params: HestonParams | None = None) -> np.ndarray:
        """Compute call prices (numpy-facing wrapper over the JAX pricer `_call_jax`).

        Inverts the characteristic function via fixed Gauss-Legendre quadrature on
        [0, config.quad.u_max]; k, t are parallel quote arrays.

        K - strike
        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        """
        params = params if params is not None else self.params
        nodes, weights = self._gauss_legendre(self.config.quad.u_max, self.config.quad.n_quad)
        prices = self._call_jax(self._pack(params), jnp.atleast_1d(jnp.asarray(k)),
                                s_0, r, jnp.atleast_1d(jnp.asarray(t)), nodes, weights)

        return np.asarray(prices)

    def put(self,
            k: float | np.ndarray,
            s_0: float,
            r: float,
            t: float | np.ndarray,
            params: HestonParams | None = None) -> float | np.ndarray:
        """Compute put prices."""
        params = params if params is not None else self.params

        return self.call(k=k, s_0=s_0, r=r, t=t, params=params) - s_0 + k * np.exp(-r * t)

    def optimize_params(self,
            k: float | np.ndarray,
            s_0: float,
            r: float,
            t: float | np.ndarray,
            call: float | np.ndarray) -> None:
        """Find optimal params.

        k -strike
        s_0 - initial price
        r - interest-rate
        t - time/maturity
        call - actual option prices
        """
        logger.info("Starting MSE optimization with initial params: %s", self.params)
        params = self.mse_optimizer(init_params=self.params, k=k, s_0=s_0, r=r, t=t, call=call)
        logger.info("Starting NUTS optimization with initial params: %s", params)
        self.params = self.nuts_optimizer(init_params=params, k=k, s_0=s_0, r=r, t=t, call=call)
        logger.info("Final calibrated params: %s", self.params)


    @staticmethod
    def _bs_call(s_0: float,
                 k: float | np.ndarray,
                 r: float,
                 t: float | np.ndarray,
                 sigma: float | np.ndarray) -> float | np.ndarray:
        """Black-Scholes call price (used only for the implied-vol inversion)."""
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s_0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t

        return s_0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)

    @staticmethod
    def _bs_vega(s_0: float,
                 k: np.ndarray,
                 r: float,
                 t: np.ndarray,
                 sigma: np.ndarray) -> np.ndarray:
        """Black-Scholes vega; used to weight price errors into ~implied-vol units."""
        sqrt_t = np.sqrt(t)
        d1 = (np.log(s_0 / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)

        return s_0 * norm.pdf(d1) * sqrt_t

    @classmethod
    def _implied_vol_scalar(cls,
                            price: float,
                            s_0: float,
                            k: float,
                            r: float,
                            t: float) -> float:
        """Invert one Black-Scholes call price to its implied volatility.

        Returns nan when the price violates the no-arbitrage bounds (it must lie
        strictly between intrinsic value and s_0), since no real implied vol
        exists there.
        """
        intrinsic = max(s_0 - k * np.exp(-r * t), 0.0)
        if price <= intrinsic or price >= s_0:
            return np.nan
        try:
            return brentq(lambda sig: cls._bs_call(s_0, k, r, t, sig) - price,
                          1e-6, 5.0, maxiter=100)
        except (ValueError, RuntimeError):
            return np.nan

    @classmethod
    def _implied_vol(cls,
                     price: np.ndarray,
                     s_0: float,
                     k: np.ndarray,
                     r: float,
                     t: np.ndarray) -> np.ndarray:
        """Vectorised Black-Scholes implied vol over parallel quote arrays."""
        price, k, t = np.broadcast_arrays(price, k, t)
        iv = [cls._implied_vol_scalar(p, s_0, kk, r, tt)
              for p, kk, tt in zip(price.ravel(), k.ravel(), t.ravel(), strict=True)]

        return np.asarray(iv, dtype=float).reshape(price.shape)

    def mse_optimizer(self,
                     init_params: HestonParams,
                     k: np.ndarray,
                     s_0: float,
                     r: float,
                     t: np.ndarray,
                     call: np.ndarray) -> HestonParams:
        """Calibrate Heston params to a whole option surface in implied-vol space.

        k, t, call are parallel arrays of equal length: each index i is one market
        quote (strike k_i, maturity t_i, observed call price call_i). A single
        5-parameter Heston vector is fit to minimize the implied-vol residuals
        across all quotes via bounded nonlinear least squares (Trust Region
        Reflective). The Feller condition is NOT enforced -- only positivity of
        the variance/speed/vol-of-vol params and |rho| < 1.

        init_params - starting point for the optimizer
        k - strikes
        s_0 - initial price
        r - interest-rate
        t - time/maturity
        call - actual (market) call prices
        """
        k = np.asarray(k, dtype=float)
        t = np.asarray(t, dtype=float)
        call = np.asarray(call, dtype=float)

        # Market implied vols are param-independent -- compute once and drop any
        # quote that has no valid implied vol (outside the no-arbitrage bounds).
        iv_market = self._implied_vol(call, s_0, k, r, t)
        mask = np.isfinite(iv_market)
        if not mask.all():
            logger.warning("Dropping %d/%d quotes outside no-arbitrage bounds.",
                           int((~mask).sum()), mask.size)
        k, t, call, iv_market = k[mask], t[mask], call[mask], iv_market[mask]
        if k.size == 0:
            msg = "No valid quotes to calibrate on."
            raise ValueError(msg)

        cfg = self.config.mse
        #         bounds are (low, high) per param in PARAM_ORDER (v_0, v_mean, a, eta, rho)
        lb = np.array(cfg.lb())
        ub = np.array(cfg.ub())

        def pack(p: HestonParams) -> np.ndarray:
            return np.array([p.v_0, p.v_mean, p.a, p.eta, p.rho])

        x0 = np.clip(pack(init_params), lb, ub)

        def residuals(x: np.ndarray) -> np.ndarray:
            cand = HestonParams(*x)
            c_model = self.call(k, s_0, r, t, params=cand)
            iv_model = self._implied_vol(c_model, s_0, k, r, t)
            res = iv_model - iv_market
            # Degenerate trial params can price below intrinsic -> nan IV; turn
            # those into a large finite penalty so the solver steers away.
            return np.nan_to_num(res, nan=cfg.nan_penalty)

        result = least_squares(residuals, x0, bounds=(lb, ub),
                               method=cfg.method, x_scale=cfg.x_scale,
                               ftol=cfg.ftol, xtol=cfg.xtol)

        fitted = HestonParams(*result.x)
        logger.info("MSE calibration: cost=%.3e status=%d feller=%s",
                    result.cost, result.status, fitted.feller())

        return fitted

    @staticmethod
    def _gauss_legendre(u_max: float, n: int) -> tuple:
        """Implement Fixed Gauss-Legendre nodes/weights mapping [-1, 1] -> [0, u_max].

        Nodes are strictly interior, so the integrand's removable 1/w singularity
        at w=0 is never evaluated. Returned as jnp arrays for the JAX pricer.
        """
        x, w = np.polynomial.legendre.leggauss(n)
        nodes = 0.5 * u_max * (x + 1.0)
        weights = 0.5 * u_max * w

        return jnp.asarray(nodes), jnp.asarray(weights)

    @staticmethod
    def _cf_jax(params_vec: jnp.ndarray,
                s_0: float,
                r: float,
                t: jnp.ndarray,
                w: jnp.ndarray | complex) -> jnp.ndarray:
        """Differentiable JAX port of the Heston characteristic function.

        params_vec - jnp array [v_0, v_mean, a, eta, rho]. Mirrors
        characteristic_function exactly (same stable (beta - h) root).
        """
        v_0, v_mean, a, eta, rho = params_vec
        alpha = - (w * w + 1j * w) / 2
        beta = a - rho * eta * 1j * w
        gamma = eta * eta / 2
        h = jnp.sqrt(beta * beta - 4 * alpha * gamma)
        g = (beta - h) / (beta + h)
        r_ = (beta - h) / (eta * eta)
        exp_ht = jnp.exp(-h * t)
        d_tw = r_ * (1 - exp_ht) / (1 - g * exp_ht)
        c_tw = a * (r_ * t - 2 * jnp.log((1 - g * exp_ht) / (1 - g)) / (eta * eta))

        return jnp.exp(c_tw * v_mean + d_tw * v_0 + 1j * w * jnp.log(s_0 * jnp.exp(r * t)))

    @classmethod
    def _call_jax(cls,
                  params_vec: jnp.ndarray,
                  k: jnp.ndarray,
                  s_0: float,
                  r: float,
                  t: jnp.ndarray,
                  nodes: jnp.ndarray,
                  weights: jnp.ndarray) -> jnp.ndarray:
        """JAX Heston call pricer over a surface via fixed Gauss-Legendre quadrature.

        Differentiable / jit-able replacement for `call`, used by NUTS. k, t are
        parallel quote arrays (N,); nodes, weights are the quadrature grid on
        [0, U] (M,). Returns call prices (N,).
        """
        t_col = t[:, None]                                          # (N, 1)
        k_col = k[:, None]                                          # (N, 1)
        w = nodes[None, :]                                          # (1, M)

        cf_den = cls._cf_jax(params_vec, s_0, r, t_col, -1j)        # (N, 1)
        cf1 = cls._cf_jax(params_vec, s_0, r, t_col, w - 1j)        # (N, M)
        cf2 = cls._cf_jax(params_vec, s_0, r, t_col, w)            # (N, M)

        phase = jnp.exp(-1j * w * jnp.log(k_col))                  # (N, M)
        integ1 = jnp.real(phase * cf1 / (1j * w * cf_den))
        integ2 = jnp.real(phase * cf2 / (1j * w))

        pi1 = 0.5 + (integ1 @ weights) / jnp.pi                     # (N,)
        pi2 = 0.5 + (integ2 @ weights) / jnp.pi

        return s_0 * pi1 - jnp.exp(-r * t) * k * pi2

    def nuts_optimizer(self,
                       init_params: HestonParams,
                       k: np.ndarray,
                       s_0: float,
                       r: float,
                       t: np.ndarray,
                       call: np.ndarray,
                       num_warmup: int | None = None,
                       num_samples: int | None = None,
                       u_max: float | None = None,
                       n_quad: int | None = None,
                       seed: int | None = None) -> HestonParams:
        """Bayesian Heston calibration over the surface via numpyro NUTS.

        Warm-started at the MSE point estimate (init_params), with priors centred
        there. The likelihood uses the differentiable JAX pricer (_call_jax) and
        vega-weighted price errors -- (model - market)/vega approximates an
        implied-vol residual to first order while staying autodiff-friendly
        (brentq inversion is not differentiable). The full posterior is stored on
        self.posterior; the posterior mean is returned as the point estimate.

        Sampling knobs (num_warmup, num_samples, u_max, n_quad, seed) fall back to
        self.config when left as None; pass them explicitly to override per call.

        k, t, call - parallel surface arrays (one market quote per index).
        """
        ncfg = self.config.nuts
        qcfg = self.config.quad
        num_warmup = ncfg.num_warmup if num_warmup is None else num_warmup
        num_samples = ncfg.num_samples if num_samples is None else num_samples
        seed = ncfg.seed if seed is None else seed
        u_max = qcfg.u_max if u_max is None else u_max
        n_quad = qcfg.n_quad if n_quad is None else n_quad

        k = np.asarray(k, dtype=float)
        t = np.asarray(t, dtype=float)
        call = np.asarray(call, dtype=float)

        # Vega weights come from the (param-independent) market implied vols; drop
        # quotes with no valid IV, exactly as in the MSE stage.
        iv_market = self._implied_vol(call, s_0, k, r, t)
        mask = np.isfinite(iv_market)
        if not mask.all():
            logger.warning("NUTS: dropping %d/%d quotes outside no-arbitrage bounds.",
                           int((~mask).sum()), mask.size)
        k, t, call, iv_market = k[mask], t[mask], call[mask], iv_market[mask]
        if k.size == 0:
            msg = "No valid quotes to calibrate on."
            raise ValueError(msg)

        vega = np.maximum(self._bs_vega(s_0, k, r, t, iv_market), ncfg.vega_floor)

        nodes, weights = self._gauss_legendre(u_max, n_quad)
        k_j = jnp.asarray(k)
        t_j = jnp.asarray(t)
        call_j = jnp.asarray(call)
        vega_j = jnp.asarray(vega)

        def model() -> None:
            scale = ncfg.lognormal_scale
            v_0 = numpyro.sample("v_0", dist.LogNormal(np.log(init_params.v_0), scale))
            v_mean = numpyro.sample("v_mean", dist.LogNormal(np.log(init_params.v_mean), scale))
            a = numpyro.sample("a", dist.LogNormal(np.log(init_params.a), scale))
            eta = numpyro.sample("eta", dist.LogNormal(np.log(init_params.eta), scale))
            rho = numpyro.sample("rho", dist.Uniform(ncfg.rho_low, ncfg.rho_high))
            sigma = numpyro.sample("sigma", dist.HalfNormal(ncfg.sigma_halfnormal_scale))

            params_vec = jnp.stack([v_0, v_mean, a, eta, rho])
            prices = self._call_jax(params_vec, k_j, s_0, r, t_j, nodes, weights)
            numpyro.sample("obs", dist.Normal(prices, sigma * vega_j), obs=call_j)

        init_vals = {"v_0": init_params.v_0, "v_mean": init_params.v_mean,
                     "a": init_params.a, "eta": init_params.eta,
                     "rho": init_params.rho, "sigma": ncfg.init_sigma}
        kernel = NUTS(model, init_strategy=init_to_value(values=init_vals))
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    progress_bar=False, num_chains=ncfg.num_chains)
        mcmc.run(jax.random.PRNGKey(seed))

        self.posterior = mcmc.get_samples()
        post_mean = HestonParams(
            v_0=float(jnp.mean(self.posterior["v_0"])),
            v_mean=float(jnp.mean(self.posterior["v_mean"])),
            a=float(jnp.mean(self.posterior["a"])),
            eta=float(jnp.mean(self.posterior["eta"])),
            rho=float(jnp.mean(self.posterior["rho"])),
        )
        logger.info("NUTS calibration: posterior mean feller=%s", post_mean.feller())

        return post_mean