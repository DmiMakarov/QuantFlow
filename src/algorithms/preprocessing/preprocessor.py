"""Option-chain cleaning and Breeden-Litzenberger marginal extraction.

Sits between the raw chain (DataLoader.load_option_chain) and the models: it turns
noisy two-sided quotes into a clean call-price surface that Heston / the neural SDE can
calibrate to, and extracts the risk-neutral terminal densities (the marginals) that are
both a Phase-1 deliverable and the Phase-2 calibration target.

Pipeline (SurfacePreprocessor.prepare):
    raw chain --clean--> tidy call-price surface --fit_svi--> per-maturity smiles
              --marginals--> risk-neutral densities mu_T(K).
"""
from dataclasses import dataclass
from logging import getLogger

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

from ..black_scholes import bs_call, bs_vega, implied_vol
from .preprocessing_config import LiquidityConfig, PreprocessConfig
from .svi import SVIParams, fit_svi_slice

logger = getLogger()


@dataclass
class Marginal:
    """Risk-neutral terminal density mu_T(K) for one maturity (Breeden-Litzenberger)."""

    ttm: float
    strikes: np.ndarray
    density: np.ndarray

    def mean(self) -> float:
        """E[S_T] under the density; should equal the forward for a martingale."""
        return float(trapezoid(self.strikes * self.density, self.strikes)
                     / trapezoid(self.density, self.strikes))


@dataclass
class PreparedSurface:
    """Everything Phase 1 needs from the option chain."""

    clean: pd.DataFrame
    svi: list[SVIParams]
    marginals: list[Marginal]


class SurfacePreprocessor:
    """Cleans an option chain and extracts its risk-neutral marginals."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """Store config; defaults reproduce the documented v1 behaviour."""
        self.config = config if config is not None else PreprocessConfig()

    # ------------------------------------------------------------------ cleaning
    @staticmethod
    def _mid_price(df: pd.DataFrame) -> pd.Series:
        """Mid of a two-sided quote, falling back to last traded price."""
        bid, ask, last = df["bid"], df["ask"], df["lastPrice"]
        two_sided = (bid > 0) & (ask > 0)

        return ((bid + ask) / 2.0).where(two_sided, last)

    @staticmethod
    def _liquidity_mask(df: pd.DataFrame, cfg: LiquidityConfig) -> pd.Series:
        """Keep only quotes with enough volume / open interest and a tight spread."""
        volume = df["volume"].fillna(0)
        open_interest = df["openInterest"].fillna(0)
        mask = (volume >= cfg.min_volume) & (open_interest >= cfg.min_open_interest)

        two_sided = (df["bid"] > 0) & (df["ask"] > 0)
        rel_spread = ((df["ask"] - df["bid"]) / df["mid"]).where(two_sided, 0.0)
        mask &= rel_spread <= cfg.max_rel_spread
        if cfg.drop_zero_bid:
            mask &= df["bid"] > 0

        return mask

    @staticmethod
    def _butterfly_keep(strike: np.ndarray, call: np.ndarray, tol: float) -> np.ndarray:
        """Boolean keep-mask dropping interior strikes where C(K) loses convexity."""
        n = strike.size
        keep = np.ones(n, dtype=bool)
        if n < 3:
            return keep
        for i in range(1, n - 1):
            slope_left = (call[i] - call[i - 1]) / (strike[i] - strike[i - 1])
            slope_right = (call[i + 1] - call[i]) / (strike[i + 1] - strike[i])
            if slope_right - slope_left < -tol:
                keep[i] = False

        return keep

    def _clean_slice(self, group: pd.DataFrame, r: float) -> pd.DataFrame:
        """Clean one maturity: OTM selection, put->call parity, IV, butterfly filter."""
        t = float(group["ttm"].iloc[0])
        forward = float(group["spot"].iloc[0]) * np.exp(r * t)
        disc = np.exp(-r * t)

        calls = group[group["type"] == "call"]
        puts = group[group["type"] == "put"]
        otm_calls = calls[calls["strike"] >= forward]
        otm_puts = puts[puts["strike"] < forward]

        strike = np.concatenate([otm_calls["strike"].to_numpy(),
                                 otm_puts["strike"].to_numpy()])
        # OTM calls keep their mid; OTM puts map to calls via parity C = P + e^{-rt}(F - K).
        call_price = np.concatenate([
            otm_calls["mid"].to_numpy(),
            otm_puts["mid"].to_numpy() + disc * (forward - otm_puts["strike"].to_numpy()),
        ])

        order = np.argsort(strike)
        strike, call_price = strike[order], call_price[order]

        iv = implied_vol(call_price, forward, strike, t, r)
        finite = np.isfinite(iv)
        if not finite.all():
            logger.warning("ttm=%.4f: dropping %d/%d quotes outside no-arbitrage bounds.",
                           t, int((~finite).sum()), finite.size)
        strike, call_price, iv = strike[finite], call_price[finite], iv[finite]

        keep = self._butterfly_keep(strike, call_price, self.config.arbitrage.butterfly_tol)
        strike, call_price, iv = strike[keep], call_price[keep], iv[keep]

        return pd.DataFrame({
            "ttm": t,
            "strike": strike,
            "forward": forward,
            "log_moneyness": np.log(strike / forward),
            "call_price": call_price,
            "iv": iv,
            "total_var": iv * iv * t,
        })

    def _apply_calendar(self, clean: pd.DataFrame) -> pd.DataFrame:
        """Drop later-maturity quotes whose total variance dips below the earlier slice."""
        tol = self.config.arbitrage.calendar_tol
        keep = pd.Series(True, index=clean.index)
        prev_k = prev_w = None
        for t in sorted(clean["ttm"].unique()):
            g = clean[clean["ttm"] == t]                 # already strike- (hence k-) sorted
            k, w = g["log_moneyness"].to_numpy(), g["total_var"].to_numpy()
            if prev_k is not None:
                within = (k >= prev_k.min()) & (k <= prev_k.max())
                violated = within & (w < np.interp(k, prev_k, prev_w) - tol)
                keep.loc[g.index[violated]] = False
            prev_k, prev_w = k, w

        return clean[keep]

    def clean(self, chain: pd.DataFrame, r: float) -> pd.DataFrame:
        """Turn a raw option chain into a tidy, arbitrage-filtered call-price surface.

        chain - raw frame from DataLoader.load_option_chain (needs strike, type, ttm,
            spot, bid, ask, lastPrice, volume, openInterest).
        r - risk-free rate; the forward is F = spot * exp(r * t) (dividends ignored in v1).
        """
        numeric = ["strike", "ttm", "spot", "bid", "ask", "lastPrice",
                   "volume", "openInterest"]
        df = chain.copy()
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        df["mid"] = self._mid_price(df)

        sane = (df["strike"] > 0) & (df["ttm"] > 0) & np.isfinite(df["mid"]) & (df["mid"] > 0)
        df = df[sane & self._liquidity_mask(df, self.config.liquidity)]
        if df.empty:
            msg = "No quotes survived liquidity/sanitation cleaning."
            raise ValueError(msg)

        slices = [self._clean_slice(df[df["ttm"] == t], r)
                  for t in sorted(df["ttm"].unique())]
        clean = pd.concat(slices, ignore_index=True)

        return self._apply_calendar(clean)

    # ------------------------------------------------------------------ SVI fit
    def fit_svi(self, clean: pd.DataFrame) -> list[SVIParams]:
        """Fit one raw-SVI smile per maturity to the cleaned surface."""
        cfg = self.config.svi
        slices = []
        for t in sorted(clean["ttm"].unique()):
            g = clean[clean["ttm"] == t]
            forward = float(g["forward"].iloc[0])
            k = g["log_moneyness"].to_numpy()
            total_var = g["total_var"].to_numpy()
            weights = None
            if cfg.vega_weight:
                # discount cancels within a slice, so r=0 keeps the weights scale-free.
                weights = bs_vega(forward, g["strike"].to_numpy(), t, g["iv"].to_numpy(), 0.0)
            fitted = fit_svi_slice(k, total_var, t, forward, weights=weights, config=cfg)
            if not fitted.is_arbitrage_free():
                logger.warning("ttm=%.4f: fitted SVI slice violates static no-arb bounds.", t)
            slices.append(fitted)

        return slices

    # ------------------------------------------------------------- marginals (BL)
    def marginals(self, svi: list[SVIParams], r: float) -> list[Marginal]:
        """Extract risk-neutral densities mu_T(K) = e^{rt} d^2C/dK^2 from the SVI smiles."""
        cfg = self.config.marginal
        out = []
        for sp in svi:
            grid = np.linspace(cfg.strike_lo * sp.forward, cfg.strike_hi * sp.forward,
                               cfg.n_grid)
            iv = np.maximum(sp.implied_vol(grid), 1e-8)        # floor avoids sigma=0 blowups
            call = bs_call(sp.forward, grid, sp.t, iv, r)
            density = np.exp(r * sp.t) * np.gradient(np.gradient(call, grid), grid)
            if cfg.kernel_bandwidth > 0:
                density = gaussian_filter1d(density, cfg.kernel_bandwidth)
            density = np.clip(density, 0.0, None)
            if cfg.normalize:
                density = density / trapezoid(density, grid)
            out.append(Marginal(ttm=sp.t, strikes=grid, density=density))

        return out

    # ------------------------------------------------------------------- orchestration
    def prepare(self, chain: pd.DataFrame, r: float) -> PreparedSurface:
        """Full pipeline: clean -> fit SVI -> extract marginals."""
        clean = self.clean(chain, r)
        svi = self.fit_svi(clean)
        margs = self.marginals(svi, r)

        return PreparedSurface(clean=clean, svi=svi, marginals=margs)
