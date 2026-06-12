"""Implementation of analytical solution of Heston Model.

Baseline in An Analysis of the Heston Stochastic Volatility Model:
Implementation and Calibration using Matlab* Ricardo Crisóstomo.
"""
import numpy as np
from scipy.integrate import quad_vec

from .heston_params import HestonParams


class HestonAnalytics():
    """Heston model implementation with analytical solution."""

    def __init__(self,
                 init_params: HestonParams) -> None:
        """Define initial Heston parameters.

        v_0 - initial variance
        v_mean - long-term variance
        a - the variance mean-reversion speed
        eta - the volatility of the variance process
        rho - correlation coefficient of Weiner processes
        """
        if(not init_params.feller()):
            msg: str = "Bad value of the params: Feller condition isnt satisfied."
            raise ValueError(msg)

        self.params = init_params

    def characteristic_function(self,
                                s_0: float,
                                r: float,
                                t: float | np.ndarray,
                                w: float | np.ndarray) -> complex | np.ndarray:
        """Calculate Heston characteristic function.

        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        alpha = - (w * w + 1j * w) / 2
        beta = self.params.a - self.params.rho * self.params.eta * 1j * w
        gamma = self.params.eta * self.params.eta / 2
        h = np.sqrt(beta * beta - 4 * alpha * gamma)
        g = (beta - h) / (beta + h)
        r_ = (beta - h) / self.params.eta / self.params.eta
        D_tw = r_ * (1 - np.exp(-h  *t)) / (1 - g * np.exp(- h * t))
        C_tw = self.params.a * (r_ * t - 2 * np.log((1 - g * np.exp(-h * t)) / (1 - g)) / self.params.eta / self.params.eta)

        return np.exp(C_tw * self.params.v_mean + D_tw * self.params.v_0 + 1j * w * np.log(s_0 * np.exp(r * t)))

    def _pi1(self,
             k: float,
             s_0: float,
             r: float,
             t: float | np.ndarray) -> float:
        """Calculate pi1.

        k - strike
        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        def integrand(w: float) -> float:
            """Define the function under integral."""
            val = np.exp(-1j * w * np.log(k))
            val *= self.characteristic_function(s_0, r, t, w - 1j)
            val /= 1j * w * self.characteristic_function(s_0, r, t, - 1j)

            return val.real

        integral, _err = quad_vec(integrand, 0, np.inf, limit=1000)

        return 0.5 + integral / np.pi

    def _pi2(self,
             k: float,
             s_0: float,
             r: float,
             t: float | np.ndarray) -> float:
        """Calculate pi2.

        K - strike
        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        def integrand(w: float) -> float:
            """Define the function under integral."""
            val = np.exp(-1j * w * np.log(k))
            val *= self.characteristic_function(s_0, r, t, w)
            val /= 1j * w

            return val.real

        integral, _err = quad_vec(integrand, 0, np.inf, limit=1000)

        return 0.5 + integral / np.pi

    def call(self,
                   k: float,
                   s_0: float,
                   r: float,
                   t: float | np.ndarray) -> float | np.ndarray:
        """Compute call price.

        Call price = S_0 * Pi_1 - e^{-rT} K Pi_2
        K - strike
        s_0 - initial price
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        pi1: float | np.ndarray = self._pi1(k, s_0, r, t)
        pi2: float | np.ndarray = self._pi2(k, s_0, r, t)

        return s_0 * pi1 - np.exp(-r * t) * pi2

    def put(self,
            k: float,
            s_0: float,
            r: float,
            t: float | np.ndarray) -> float | np.ndarray:
        """Compute put prices."""
        return self.call - s_0 + k * np.exp(-r * t)

    def optimize_params(self,
            k: float,
            s_0: float,
            r: float,
            t: float | np.ndarray) -> None:
        pass

    def mse_optmizer(self,
                     init_params: HestonParams,
                     k: float,
                     s_0: float,
                     r: float,
                     t: float | np.ndarray) -> HestonParams:
        pass

    def nuts_optimizer(self,
                       init_params: HestonParams,
                       k: float,
                       s_0: float,
                       r: float,
                       t: float | np.ndarray) -> HestonParams:
        pass