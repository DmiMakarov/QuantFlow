"""Tests for the put-call-parity implied rate.

The key property is exactness: if quotes are BUILT from parity at a known rate, the
regression must recover that rate to machine precision -- there is no fitting error to
tolerate, only the linear algebra.
"""
import numpy as np
import pandas as pd
import pytest

from src.algorithms.preprocessing.rates import implied_rate, implied_rate_curve

SPOT = 100.0


def _parity_chain(rate: float, ttms: tuple[float, ...] = (0.5,),
                  strikes: tuple[float, ...] = (90.0, 100.0, 110.0),
                  spread: float = 1.0) -> pd.DataFrame:
    """A chain whose call/put mids satisfy parity EXACTLY at `rate`.

    C - P = S - K e^{-rT} is all the regression sees, so the call leg is free -- but it must
    be built off the DISCOUNTED intrinsic, or deep-ITM calls imply a negative put mid and the
    two-sided filter (rightly) throws the strike away before it reaches the regression.
    bid/ask straddle each mid so the filter passes and the mid is recovered exactly.
    """
    rows = []
    for ttm in ttms:
        for strike in strikes:
            call_mid = max(SPOT - strike * np.exp(-rate * ttm), 0.0) + 5.0
            put_mid = call_mid - SPOT + strike * np.exp(-rate * ttm)
            for kind, mid in (("call", call_mid), ("put", put_mid)):
                rows.append({"expiry": f"exp-{ttm}", "type": kind, "strike": strike,
                             "ttm": ttm, "bid": mid - spread / 2, "ask": mid + spread / 2})

    return pd.DataFrame(rows)


def test_implied_rate_recovers_the_rate_it_was_built_from() -> None:
    chain = _parity_chain(rate=0.0573)
    assert implied_rate(chain) == pytest.approx(0.0573, abs=1e-12)


def test_implied_rate_curve_is_per_expiry() -> None:
    chain = _parity_chain(rate=0.05, ttms=(0.25, 1.0, 2.0))
    curve = implied_rate_curve(chain)

    assert list(curve["ttm"]) == [0.25, 1.0, 2.0]          # sorted by ttm
    assert curve["rate"].to_numpy() == pytest.approx(0.05, abs=1e-12)
    assert (curve["n_quotes"] == 3).all()


def test_implied_rate_is_the_median_across_expiries() -> None:
    """One expiry with a wild rate must not drag the surface's rate with it -- that is
    the whole reason this takes a median rather than a mean."""
    good = _parity_chain(rate=0.05, ttms=(0.5, 1.0))
    bad = _parity_chain(rate=0.90, ttms=(2.0,))            # a crossed/stale expiry
    chain = pd.concat([good, bad], ignore_index=True)

    assert implied_rate(chain) == pytest.approx(0.05, abs=1e-12)


def test_expiry_with_too_few_matched_strikes_is_skipped() -> None:
    chain = _parity_chain(rate=0.05, ttms=(1.0,), strikes=(90.0, 110.0))   # only 2
    assert implied_rate_curve(chain).empty


def test_one_sided_quotes_are_excluded() -> None:
    """A zero bid means no real market; such a strike must not enter the regression."""
    chain = _parity_chain(rate=0.05, ttms=(1.0,), strikes=(90.0, 100.0, 110.0))
    chain.loc[chain["strike"] == 90.0, "bid"] = 0.0        # knocks out both legs at K=90

    assert implied_rate_curve(chain).empty                 # 2 matched strikes left -> skipped


def test_non_positive_discount_factor_is_skipped() -> None:
    """An upward-sloping C - P implies disc <= 0, which has no real rate."""
    chain = _parity_chain(rate=0.05, ttms=(1.0,))
    calls = chain["type"] == "call"
    chain.loc[calls, ["bid", "ask"]] = chain.loc[calls, ["bid", "ask"]].to_numpy() \
        + 10.0 * chain.loc[calls, "strike"].to_numpy()[:, None]

    assert implied_rate_curve(chain).empty


def test_falls_back_to_default_and_warns_when_no_expiry_is_usable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chain = _parity_chain(rate=0.05, ttms=(1.0,), strikes=(100.0,))   # 1 strike -> unusable
    with caplog.at_level("WARNING"):
        assert implied_rate(chain, default=0.037) == 0.037
    assert "two-sided quotes" in caplog.text


def test_zero_ttm_expiry_is_skipped() -> None:
    """Expired contracts divide by T; they must be dropped, not turned into inf."""
    chain = _parity_chain(rate=0.05, ttms=(0.0,))
    assert implied_rate_curve(chain).empty
