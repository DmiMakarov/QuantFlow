"""Implementation of analytical solution of Heston Model.

Baseline in An Analysis of the Heston Stochastic Volatility Model:
Implementation and Calibration using Matlab* Ricardo Crisóstomo.
"""
import numpy as np

from src.algorithms.heston.heston_params import HestonParams

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

    def charasteristic_function(self, 
                                s_0: float,
                                vvol: float,
                                r: float,
                                t: float | np.ndarray,
                                w: float | np.ndarray) -> float | np.ndarray:
        """Calculate Heston characteristic function.

        s_0 - initial price
        vvol - volatility of the variance process
        r - risk-free rate
        t - time/maturity
        w - points at which to evaluate the function
        """
        alpha = - w*w/2 - 

    def _pi1(self):
        pass

    def _pi2(self):
        pass

    def call_price(self):
        pass

    def put_price(self):
        pass

    def optimize_params(self):
        pass