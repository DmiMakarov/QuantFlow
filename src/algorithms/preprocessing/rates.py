"""Risk-free rate implied by the option chain itself, via put-call parity.

The rate is the one input SurfacePreprocessor.prepare(chain, r) needs that does not come
from the chain's own columns, and it is not observable in the yfinance feed. Rather than
hard-coding a treasury yield (which ignores the dividend stream and the repo rate baked
into the quotes), back it out of the quotes: parity forces

    C - P = e^{-rT} (F - K)   =>   d(C - P)/dK = -e^{-rT},

so a straight line through the call-put mid spread against strike recovers the discount
factor for that expiry, with dividends already folded into F.

Lives in preprocessing (not eval) because it produces a *market-data* input to the
calibration, not a metric.
"""
from logging import getLogger

import numpy as np
import pandas as pd

logger = getLogger()

MIN_PARITY_QUOTES: int = 3   # a slope through fewer than 3 matched strikes is noise


def implied_rate_curve(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry risk-free rate from put-call parity.

    For each expiry, matches calls to puts on strike over two-sided quotes only (a missing
    bid means no real market, and a stale lastPrice would bias the slope), regresses the
    mid spread C - P on K, and turns the slope into a rate via disc = -slope, r = -ln(disc)/T.
    Expiries with fewer than MIN_PARITY_QUOTES matched strikes, or a non-positive implied
    discount factor, are skipped.

    The SPREAD of these rates across expiries is worth looking at directly: it measures how
    much a single flat r misprices the forward, which is a leading suspect whenever the
    Breeden-Litzenberger marginals fail to recover the forward as their mean.

    Returns a frame with columns (ttm, rate, n_quotes), sorted by ttm. Empty if no expiry
    has enough matched two-sided quotes.
    """
    rows = []
    calls, puts = chain[chain["type"] == "call"], chain[chain["type"] == "put"]
    for expiry, cg in calls.groupby("expiry"):
        pg = puts[puts["expiry"] == expiry]
        matched = cg.merge(pg, on="strike", suffixes=("_c", "_p"))
        two_sided = matched[(matched["bid_c"] > 0) & (matched["ask_c"] > 0)
                            & (matched["bid_p"] > 0) & (matched["ask_p"] > 0)]
        if len(two_sided) < MIN_PARITY_QUOTES:
            continue

        ttm = float(two_sided["ttm_c"].iloc[0])
        mid_c = (two_sided["bid_c"] + two_sided["ask_c"]) / 2
        mid_p = (two_sided["bid_p"] + two_sided["ask_p"]) / 2
        slope = np.polyfit(two_sided["strike"], mid_c - mid_p, 1)[0]

        disc = -slope                                  # e^{-rT}
        if disc <= 0 or ttm <= 0:                      # crossed/degenerate quotes
            continue
        rows.append({"ttm": ttm, "rate": -np.log(disc) / ttm, "n_quotes": len(two_sided)})

    return pd.DataFrame(rows, columns=["ttm", "rate", "n_quotes"]).sort_values("ttm",
                                                                              ignore_index=True)


def implied_rate(chain: pd.DataFrame, default: float = 0.04) -> float:
    """Single flat risk-free rate for the chain: the median of implied_rate_curve.

    The median (not the mean) because a single expiry with a crossed or stale quote can
    throw its regression far off, and one bad expiry should not move the rate the whole
    surface is calibrated under. Falls back to `default` when no expiry yields a usable
    slope -- a flat 4% is a better answer than a crash on a thin chain.
    """
    curve = implied_rate_curve(chain)
    if curve.empty:
        logger.warning("No expiry had enough two-sided quotes for parity; using r=%.4f.",
                       default)
        return default

    return float(curve["rate"].median())
