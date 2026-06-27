"""Tests for SurfacePreprocessor: cleaning filters and Breeden-Litzenberger marginals."""
import logging

import numpy as np
import pandas as pd
import pytest

from src.algorithms.black_scholes import bs_call
from src.algorithms.preprocessing import (
    MarginalConfig,
    PreprocessConfig,
    SurfacePreprocessor,
    SVIConfig,
)


def make_chain(sigma: float = 0.2, spot: float = 100.0, r: float = 0.0,
               maturities: tuple[float, ...] = (0.1, 0.5, 1.0),
               strikes: np.ndarray | None = None) -> pd.DataFrame:
    """Synthetic, arbitrage-free option chain priced at a flat Black-Scholes vol.

    Emits both a call and a (parity-consistent) put at every strike/maturity with tight
    two-sided quotes and ample liquidity, mimicking the columns DataLoader produces.
    """
    if strikes is None:
        strikes = np.linspace(70.0, 130.0, 13)
    rows = []
    for t in maturities:
        forward = spot * np.exp(r * t)
        disc = np.exp(-r * t)
        for strike in strikes:
            call = float(bs_call(forward, strike, t, sigma, r))
            put = call - disc * (forward - strike)
            for kind, price in (("call", call), ("put", put)):
                rows.append({
                    "strike": strike, "type": kind, "ttm": t, "spot": spot,
                    "lastPrice": price, "bid": price * 0.99, "ask": price * 1.01,
                    "volume": 100, "openInterest": 100,
                })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------- mid price
def test_mid_price_uses_mid_then_last() -> None:
    df = pd.DataFrame({
        "bid": [9.0, 0.0], "ask": [11.0, 0.0], "lastPrice": [10.5, 7.0],
    })
    mid = SurfacePreprocessor._mid_price(df)
    assert mid.iloc[0] == 10.0       # two-sided -> midpoint
    assert mid.iloc[1] == 7.0        # one-sided -> last traded price


# --------------------------------------------------------------------- liquidity
def test_liquidity_mask_drops_each_violation() -> None:
    df = pd.DataFrame({
        "bid": [9.0, 9.0, 9.0, 1.0, 0.0],
        "ask": [11.0, 11.0, 11.0, 19.0, 11.0],
        "mid": [10.0, 10.0, 10.0, 10.0, 5.5],
        "volume": [100, 0, 100, 100, 100],
        "openInterest": [100, 100, 0, 100, 100],
    })
    mask = SurfacePreprocessor._liquidity_mask(df, PreprocessConfig().liquidity).to_numpy()
    # row 0 good; 1 no volume; 2 no OI; 3 spread too wide; 4 zero bid.
    assert mask.tolist() == [True, False, False, False, False]


def test_liquidity_mask_keeps_one_sided_when_zero_bid_allowed() -> None:
    cfg = PreprocessConfig().liquidity
    cfg.drop_zero_bid = False
    df = pd.DataFrame({
        "bid": [0.0], "ask": [11.0], "mid": [7.0],
        "volume": [100], "openInterest": [100],
    })
    # one-sided quote: spread test is skipped, so it survives when zero bids are allowed.
    assert SurfacePreprocessor._liquidity_mask(df, cfg).to_numpy().tolist() == [True]


# --------------------------------------------------------------------- butterfly
def test_butterfly_keep_short_slice_is_all_kept() -> None:
    keep = SurfacePreprocessor._butterfly_keep(np.array([90.0, 100.0]),
                                               np.array([12.0, 5.0]), 1e-8)
    assert keep.tolist() == [True, True]


def test_butterfly_keep_flags_non_convex_point() -> None:
    strike = np.array([90.0, 100.0, 110.0, 120.0])
    convex = np.array([12.0, 6.0, 3.0, 1.5])
    assert SurfacePreprocessor._butterfly_keep(strike, convex, 1e-8).tolist() == \
        [True, True, True, True]
    bumped = np.array([12.0, 6.0, 7.0, 1.5])           # the K=110 point breaks convexity
    assert SurfacePreprocessor._butterfly_keep(strike, bumped, 1e-8).tolist() == \
        [True, True, False, True]


# --------------------------------------------------------------------- calendar
def test_apply_calendar_drops_total_variance_dip() -> None:
    clean = pd.DataFrame({
        "ttm": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
        "log_moneyness": [-0.1, 0.0, 0.1, -0.1, 0.0, 0.1, 0.5],
        "total_var": [0.05, 0.04, 0.05, 0.10, 0.03, 0.10, 0.20],
    })
    out = SurfacePreprocessor()._apply_calendar(clean)
    # the t=1 quote at k=0 (w=0.03 < earlier 0.04) is removed; the out-of-range k=0.5 kept.
    kept = out[out["ttm"] == 1.0]["log_moneyness"].tolist()
    assert 0.0 not in kept
    assert 0.5 in kept
    assert len(out) == 6


# --------------------------------------------------------------------- clean
def test_clean_returns_tidy_surface() -> None:
    clean = SurfacePreprocessor().clean(make_chain(), r=0.0)
    assert not clean.empty
    assert set(clean.columns) == {
        "ttm", "strike", "forward", "log_moneyness", "call_price", "iv", "total_var",
    }
    assert np.isfinite(clean["iv"]).all()
    assert (clean["iv"] > 0).all()
    # recovered IV should be close to the 0.2 we priced at.
    assert abs(clean["iv"].mean() - 0.2) < 0.02


def test_clean_drops_quotes_outside_no_arb_bounds(caplog: pytest.LogCaptureFixture) -> None:
    chain = make_chain(maturities=(0.5,))
    bad = chain.iloc[[0]].copy()
    bad["strike"] = 120.0                # OTM call...
    bad["type"] = "call"
    bad[["lastPrice", "bid", "ask"]] = 100.0   # ...priced at/above the upper bound
    chain = pd.concat([chain, bad], ignore_index=True)

    with caplog.at_level(logging.WARNING):
        clean = SurfacePreprocessor().clean(chain, r=0.0)

    assert any("no-arbitrage bounds" in rec.message for rec in caplog.records)
    assert (clean["call_price"] < 100.0).all()


def test_clean_raises_when_everything_filtered() -> None:
    chain = make_chain(maturities=(0.5,))
    chain[["volume", "openInterest"]] = 0
    with pytest.raises(ValueError, match="No quotes survived"):
        SurfacePreprocessor().clean(chain, r=0.0)


# --------------------------------------------------------------------- SVI
def test_fit_svi_returns_one_slice_per_maturity() -> None:
    pre = SurfacePreprocessor()
    clean = pre.clean(make_chain(), r=0.0)
    svi = pre.fit_svi(clean)
    assert len(svi) == clean["ttm"].nunique()
    assert all(sp.is_arbitrage_free() for sp in svi)


def test_fit_svi_without_vega_weight() -> None:
    cfg = PreprocessConfig()
    cfg.svi.vega_weight = False
    pre = SurfacePreprocessor(cfg)
    clean = pre.clean(make_chain(), r=0.0)
    assert len(pre.fit_svi(clean)) == clean["ttm"].nunique()


def test_fit_svi_warns_on_arbitrage_violation(caplog: pytest.LogCaptureFixture) -> None:
    clean = SurfacePreprocessor().clean(make_chain(maturities=(0.5,)), r=0.0)
    # bounds that force a strongly negative level -> the fitted slice cannot be arb-free.
    bad = PreprocessConfig(svi=SVIConfig(bounds=(
        (-1.0, -0.9), (1e-4, 1e-3), (-0.5, 0.5), (-2.0, 2.0), (1e-4, 1e-3),
    )))
    with caplog.at_level(logging.WARNING):
        svi = SurfacePreprocessor(bad).fit_svi(clean)
    assert not svi[0].is_arbitrage_free()
    assert any("static no-arb bounds" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------- marginals
def test_marginals_are_valid_densities_recovering_the_forward() -> None:
    pre = SurfacePreprocessor()
    surface = pre.prepare(make_chain(), r=0.0)
    assert len(surface.marginals) == len(surface.svi)
    for marg in surface.marginals:
        assert np.all(marg.density >= 0)
        integral = np.trapezoid(marg.density, marg.strikes)
        assert abs(integral - 1.0) < 1e-6           # normalised
        assert abs(marg.mean() - 100.0) < 1.0        # martingale: E[S_T] ~= forward


def test_marginals_with_kernel_smoothing() -> None:
    cfg = PreprocessConfig(marginal=MarginalConfig(kernel_bandwidth=2.0))
    pre = SurfacePreprocessor(cfg)
    surface = pre.prepare(make_chain(maturities=(0.5,)), r=0.0)
    marg = surface.marginals[0]
    assert np.all(np.isfinite(marg.density))
    assert abs(marg.mean() - 100.0) < 1.5


def test_marginals_without_normalisation() -> None:
    cfg = PreprocessConfig(marginal=MarginalConfig(normalize=False))
    pre = SurfacePreprocessor(cfg)
    clean = pre.clean(make_chain(maturities=(0.5,)), r=0.0)
    marg = pre.marginals(pre.fit_svi(clean), r=0.0)[0]
    # an un-normalised lognormal-ish density integrates to ~1 only by construction here,
    # so just assert it is a finite, non-negative curve.
    assert np.all(marg.density >= 0)
    assert np.all(np.isfinite(marg.density))


# --------------------------------------------------------------------- end to end
def test_prepare_returns_full_surface() -> None:
    surface = SurfacePreprocessor().prepare(make_chain(), r=0.0)
    n_mat = surface.clean["ttm"].nunique()
    assert n_mat == 3
    assert len(surface.svi) == n_mat
    assert len(surface.marginals) == n_mat
