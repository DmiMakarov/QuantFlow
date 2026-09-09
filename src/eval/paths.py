"""The scenario contract every generative model in the project emits.

PROJECT_PLAN.md §8's cardinal rule is that every method is evaluated identically. That only
holds if the evaluation code cannot tell which model produced the paths it is scoring -- so
models do not hand `eval` a Heston object or a torch model, they hand it a `Paths`.

Deliberately numpy-only and dependency-free: `src/algorithms/heston/heston_mc.py` imports it
today and Phase 2's neural SDE will import it too, so it must be cheap to depend on. Note the
direction -- models depend on eval's contract, eval never depends on a model.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TIME_TOL: float = 1e-9   # maturities are matched to the grid within this absolute tolerance


@dataclass
class Paths:
    """Simulated risk-neutral paths on a shared time grid.

    times    - (n_steps + 1,) strictly increasing, times[0] == 0. Contains every requested
               maturity EXACTLY, so a single bundle serves the whole surface: slice it per
               maturity rather than re-simulating.
    spot     - (n_paths, n_steps + 1) simulated S_t.
    variance - (n_paths, n_steps + 1) instantaneous v_t, or None when not retained.
    r        - the risk-free rate the paths were simulated under (needed to discount).
    """

    times: np.ndarray
    spot: np.ndarray
    variance: np.ndarray | None = None
    r: float = 0.0

    @property
    def s_0(self) -> float:
        """The (common) initial spot."""
        return float(self.spot[0, 0])

    @property
    def n_paths(self) -> int:
        """Number of simulated paths."""
        return int(self.spot.shape[0])

    @property
    def terminal(self) -> np.ndarray:
        """(n_paths,) spot at the final time on the grid."""
        return self.spot[:, -1]

    def index_of(self, t: float) -> int:
        """Column index of maturity `t` on the time grid.

        Raises ValueError when `t` is not on the grid: silently snapping to the nearest
        column would price a different maturity than the caller asked for, and the error
        would surface as an unexplained bias in the pricing metrics rather than a crash.
        """
        hit = np.flatnonzero(np.abs(self.times - t) <= TIME_TOL)
        if hit.size == 0:
            msg = (f"maturity {t} is not on the simulated grid "
                   f"[{self.times[0]}, {self.times[-1]}]; simulate it explicitly.")
            raise ValueError(msg)

        return int(hit[0])

    def slice_at(self, t: float) -> np.ndarray:
        """(n_paths,) simulated S_t at maturity `t`."""
        return self.spot[:, self.index_of(t)]

    def samples(self, ttms: Sequence[float]) -> dict[float, np.ndarray]:
        """{ttm: S_ttm} -- the format every distributional metric consumes.

        This is the seam that keeps metrics.py decoupled from Paths entirely: the metrics
        take a plain dict of arrays, so they are testable with hand-built numbers and
        Phase 2's torch model can feed them with a single .numpy() call.
        """
        return {float(t): self.slice_at(t) for t in ttms}
