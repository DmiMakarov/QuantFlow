"""Class for Heston model params."""
from dataclasses import dataclass


@dataclass
class HestonParams:
    """Risk-neutral Heston params.

    v_0 - initial variance
    v_mean - long-term variance
    a - the variance mean-reversion speed
    eta - the volatility of the variance process
    rho - correlation coefficient of Weiner processes
    """

    v_0: float
    v_mean: float
    a: float
    eta: float
    rho: float

    def feller(self) -> bool:
        """Whether the Feller condition 2*a*v_mean >= eta**2 holds.

        When True, the variance process v_t stays strictly positive and never
        touches zero. A violation does not affect the analytical pricer (the
        characteristic function handles it), but the MC simulator then needs a
        positivity-preserving scheme (e.g. full truncation). Real equity-index
        calibrations routinely violate it -- treat as a diagnostic, not a reject.
        """
        return 2.0 * self.a * self.v_mean >= self.eta**2

