# QuantFlow — Project Plan v2

> **QuantFlow** — *Generative market models for option-chain calibration and deep hedging, on equity index (SPX/VIX) and crypto (BTC/ETH).*
>
> Author: Dmitriy (Skoltech, ex-MIPT)
> Status: Draft v0.4 — Phase 2 re-specified after the Phase 1 review (2026-09-07)
> License (intended): Apache-2.0 for the library, CC-BY-SA 4.0 for the notebooks and writeups.

> **Changelog v0.3 → v0.4.** Flow matching is no longer the v1.0 engine (§1.3 explains why it
> cannot be); v1.0 uses a Dupire local-vol baseline and a **neural local-stochastic-volatility (LSV)**
> model calibrated by a price-space loss. torchsde and adjoint training are dropped. Deep hedging is
> re-specified (pathwise gradients, transaction costs, cross-model evaluation). v2.0 starts from
> martingale Sinkhorn and Light Schrödinger Bridge rather than IPF with score networks. v3.0 is
> repositioned: neural joint SPX/VIX calibration has been demonstrated since this plan was written.
> Phase 1 numbers corrected to the benchmark table. Dead scaffolding sections removed.

---

## 0. TL;DR

Build the open-source library that does what every bank's exotics desk is privately rebuilding: a **generative model of arbitrage-consistent market dynamics** calibrated to option chains — first as a neural local-stochastic-volatility model, then upgraded to Schrödinger bridges — with **deep hedging** as the downstream evaluation.

**Restructured for honest execution.** This project is split into four releases of increasing ambition. Each release is independently valuable — partial completion still produces a real artifact. The original v1 plan tried to deliver everything in 14 weeks; this v2 plan acknowledges that v1 alone is a 5-6 month part-time project, and structures the rest accordingly.

| Release | Scope | Realistic timeline | Career-relevant value |
|---|---|---|---|
| **v1.0** | SPX calibration via neural LSV + Dupire and Heston baselines + basic deep hedging + clean benchmark | 5-6 months | Strong portfolio piece. Sufficient for interviews at Tier 2 quant firms and crypto market makers. |
| **v2.0** | Schrödinger bridge upgrade (martingale Sinkhorn → LightSB → DSBM) + Henry-Labordère martingale projection | +2-3 months | Adds research depth. Sufficient for ICAIF / NeurIPS ML4Finance workshop submission. |
| **v3.0** | Joint SPX/VIX calibration: open-source reproduction + SB-based model selection | +3 months | Genuine research contribution if it works; informative negative result if it doesn't. |
| **v4.0** | Crypto extension (BTC/ETH on Deribit) + Streamlit demo + PyPI release | +2-3 months | Productized library. Useful for Dubai crypto market maker applications. |

**Realistic total timeline: 12-18 months at part-time pace.** Not 14 weeks.

The core insight: this is one math-heavy project that simultaneously serves three goals — a deep learning artifact extending the author's Skoltech work, a portfolio piece for quant interviews, and the technical kernel of something that could become a library or service. The path to all three is the same; the only question is how far along the path the project goes.

---

## 1. The problem, precisely

### 1.1 Setup

Let $S_t$ be a (positive) underlying price process under the risk-neutral measure $\mathbb Q$. At time $t=0$ the market quotes European call prices $C^{\text{mkt}}(K, T)$ for a finite grid of strikes $K$ and maturities $T \in \{T_1, \ldots, T_M\}$. Equivalent specifications: implied vol surface $\sigma^{\text{impl}}(K,T)$ (Black–Scholes inversion) or the **Breeden–Litzenberger terminal densities**

$$\mu_{T_i}(K) \;=\; e^{rT_i}\,\frac{\partial^2 C^{\text{mkt}}(K,T_i)}{\partial K^2}, \qquad i=1,\ldots,M.$$

After cleaning (no-butterfly arbitrage, calendar arbitrage, illiquid wings) and a small amount of smoothing/extrapolation, the $\mu_{T_i}$ are the *risk-neutral marginal distributions of $S$* at the traded maturities.

### 1.2 Goal

Find a continuous-time stochastic process $(S_t)_{t \in [0,T_M]}$, specified by an SDE

$$dS_t \;=\; r\,S_t\,dt \;+\; S_t\,\sigma_\theta(t, S_t, V_t)\,dW_t, \qquad dV_t \;=\; b(t, V_t)\,dt + a(t, V_t)\,dB_t,$$

with $W,B$ correlated Brownian motions, $\sigma_\theta$ (partly) a neural network, such that:

1. **Marginal match**: simulated marginals $\mathrm{Law}_\mathbb{Q}(S_{T_i})$ equal $\mu_{T_i}$ for all $i$.
2. **No-arbitrage**: discounted $S_t$ is a martingale under $\mathbb Q$; the drift is structurally pinned at $r$.
3. **Joint surface match** (v3 only): simulated VIX$_t$ trajectories give VIX options prices that match the VIX surface simultaneously.
4. **Smoothness / regularity**: implied dynamics not pathological (no exploding variance), reproduce stylised facts (vol clustering, leverage effect, fat tails).

### 1.3 Why this is hard — and what the vanillas do *not* tell you

Conditions (1) and (2) together — finding any martingale with prescribed marginals — has a unique solution only up to dynamics; this is the **multi-marginal martingale optimal-transport problem**. Classical SDE classes (Heston, SABR, rough Bergomi) parametrize a small slice of the solution space and so generically miss either some marginals or the dependence structure required for (3). Neural SDEs (Cuchiero et al. 2020, Gierjatowicz et al. 2020, Cohen et al. 2023) expand the function class but still leave the optimization problem under-specified.

**Two facts that shape every design choice below.**

*Fact 1 — the vanilla surface is matched exactly by local volatility (Dupire 1994) and does not identify anything beyond it.* Given an arbitrage-free surface, $\sigma_{\text{loc}}^2(T,K) = \partial_T C / (\tfrac12 K^2 \partial_{KK} C)$ reproduces every $\mu_{T_i}$ and is a martingale by construction. Infinitely many other martingale models share the same marginals; they differ in forward smiles, exotic prices and hedges. So "matches the marginals within tolerance" is not a differentiating criterion — Dupire is the ceiling for every marginal metric — and the benchmark must include at least one **dynamics** metric that vanillas do not pin (§8). By Gyöngy (1986), *any* Itô process $dS = S\sigma_t dW$ has marginals reproduced by the local vol $\sigma_{\text{loc}}^2(t,x) = \mathbb E[\sigma_t^2 \mid S_t = x]$, which is what lets a stochastic-vol model be bent onto the vanillas via a leverage function (§2.2).

*Fact 2 — a deterministic transport between two distinct marginals is never a martingale coupling.* If $S_{T_2} = f(S_{T_1})$ and $\mathbb E[S_{T_2}\mid S_{T_1}] = S_{T_1}$ then $f = \mathrm{id}$ and the marginals coincide. Any ODE flow between consecutive maturities — flow matching, rectified flow, an OT map — therefore violates (2) by construction; in 1D the OT map is the comonotone quantile map, the coupling *furthest* from martingale. In addition $\mu_0 = \delta_{S_0}$ is a point mass, which no flow ODE can spread. **This is why v0.3's "flow-matching warm-start" was dropped:** it targeted the wrong object. Flow/bridge *matching* reappears in v2.0 where it belongs — as the regression step of Schrödinger-bridge solvers, which are stochastic and can be martingale.

**v1.0 approach: neural LSV.** Take the Phase 1 Heston posterior as the prior dynamics of the variance factor and learn a leverage function $L_\theta(t, S)$ so that $\sigma = L_\theta \sqrt{V}$ hits the vanillas. Calibrated directly on option prices by backpropagation through a Monte-Carlo simulator. Well-conditioned, cheap, and the only neural-SDE calibration recipe with a track record on real vanilla surfaces.

**v2.0 upgrade: Schrödinger bridge for principled selection.** Among all martingale measures with the prescribed marginals, pick the one of minimum relative entropy to a reference martingale measure. This is the unique entropic-OT solution: the most likely arbitrage-consistent market dynamics given the option chain. v1.0 answers "find *a* calibrated LSV"; v2.0 answers "find *the* one closest to the reference".

### 1.4 Downstream evaluation

A market model is only as good as the decisions it informs.

- **Exotic pricing**: MC prices for autocallables, barriers, variance swaps, forward-start options, cliquets. Compare vs. street consensus where available and vs. Heston / rough Bergomi.
- **Deep hedging** (Buehler et al. 2019): train a hedging policy on scenarios from the market model; evaluate replication P&L on held-out historical paths *and* on scenarios from the competing models. Lowest CVaR wins.
- **Robustness**: $\pm 5\%$ surface bump; report price/hedge sensitivity.

---

## 2. The mathematical engine

Different releases use different mathematical engines. This section describes them in the order they appear in the codebase.

### 2.1 Local volatility and the Gyöngy projection (v1.0)

With total implied variance $w(T,k) = \sigma^{\text{impl}}(T,k)^2 T$ in log-moneyness $k = \log K/F_T$, Dupire's formula reads (Gatheral 2006)

$$\sigma_{\text{loc}}^2(T,k) \;=\; \frac{\partial_T w}{\,1 - \tfrac{k}{w}\partial_k w + \tfrac14\!\left(-\tfrac14 - \tfrac1w + \tfrac{k^2}{w^2}\right)(\partial_k w)^2 + \tfrac12 \partial_{kk} w\,}.$$

The SVI slices from Phase 1 give $w$ and its $k$-derivatives analytically; $w$ is interpolated linearly in $T$ between slices (calendar-consistent); $k$ is clamped to the traded range (flat extrapolation — the same lesson as the BL grid). A log-Euler Monte Carlo of $dX = -\tfrac12\sigma_{\text{loc}}^2 dt + \sigma_{\text{loc}} dW$, $X = \log S/F$, produces `Paths` scored by the shared harness. This is the **marginal-matching ceiling** and the second classical baseline row.

For any stochastic-vol process, Gyöngy's theorem gives the leverage function that projects it onto the same marginals:

$$L^2(t,x) \;=\; \frac{\sigma_{\text{loc}}^2(t,x)}{\mathbb E[V_t \mid S_t = x]}.$$

The particle method (Guyon & Henry-Labordère 2012) solves this fixed point by kernel regression on simulated paths; it is the production LSV calibration algorithm. §2.2 is its neural cousin.

### 2.2 Neural LSV realization (v1.0)

State $(X_t, V_t)$ with $X = \log S_t/F_t$ (so the drift is pinned architecturally and the discrete scheme is an exact martingale step):

$$dX_t = -\tfrac12 \sigma_t^2\,dt + \sigma_t\,dW_t, \qquad \sigma_t = L_\theta(t, X_t)\sqrt{V_t}, \qquad dV_t = \kappa(\bar v - V_t)\,dt + \eta\sqrt{V_t}\,dB_t, \quad d\langle W,B\rangle = \rho\,dt.$$

- $L_\theta$: small MLP $(t/T_{\max}, x) \to 64 \to 64 \to$ softplus, last layer zero-initialised so $L \equiv 1$ at the start — the model *begins* as the Phase 1 Heston and the optimiser only bends the leverage.
- $(v_0, \kappa, \bar v, \eta, \rho)$: initialised from the Phase 1 NUTS posterior mean; reparametrised (softplus / tanh) and optionally learnable.
- Simulation: hand-written full-truncation Euler in PyTorch, float64, CPU, on the same maturity-exact time grid as the Heston simulator. **No torchsde, no adjoint**: for 1D, 8 maturities and ~260 steps, plain backprop through the solver is exact and faster; the stochastic adjoint is a memory optimisation with worse gradients. QE is not used because its branch switching is not gradient-friendly.
- Loss: Monte-Carlo prices of the *train* quotes (wing split, §8), $\sum_j w_j \big(C^{\text{MC}}_j(\theta) - C^{\text{mkt}}_j\big)^2 + \lambda\,\mathbb E[(L_\theta - 1)^2]$, with $w_j = 1/\text{vega}_j^2$ so the loss is first-order an implied-vol error, and the regulariser shrinking to Heston wherever data is absent. **Not** a Wasserstein loss against the BL densities: those are second derivatives of a smoothed interpolant, truncated to the traded range (`tail_mass` up to 8.6% on the June snapshot), and W1 in price units under-weights the ATM region. Wasserstein stays a *reported metric*.

### 2.3 Schrödinger bridges (v2.0)

Given $\mu_0, \mu_T$ and reference path measure $\mathbb W$ on $C([0,T];\mathbb R^d)$, the **Schrödinger problem** is

$$\mathbb P^\star = \arg\min_{\mathbb P}\ \mathrm{KL}(\mathbb P \,\|\, \mathbb W) \quad \text{s.t. } \mathbb P_0 = \mu_0,\ \mathbb P_T = \mu_T.$$

Léonard (2014) is canonical. Solved computationally by **Sinkhorn on path space** (IPF), each half-step being a diffusion bridge regression implementable as score matching (De Bortoli et al. 2021; Shi et al. 2023 — DSBM makes IPF provably convergent without bias).

For v2.0's actual problem — one dimension, eight marginals, a martingale constraint — there are exact and cheap solvers that should come first:

1. **Martingale Sinkhorn on a strike grid** (De March & Henry-Labordère 2019; Henry-Labordère 2019). Entropic martingale OT between consecutive BL marginals, Sinkhorn iterations with the martingale constraint, no neural networks, seconds to converge. Its output is precisely an LSV-type model: v2.0 *selects which* LSV, continuous with v1.0.
2. **Light Schrödinger Bridge / LightSB-M** (Korotin, Gushchin, Burnaev 2024; Gushchin et al. 2024 — Skoltech/AIRI). Closed-form Gaussian-mixture Schrödinger potentials, no IPF. The natural neural step up.
3. **DSBM / IPF with score networks** only when the state is $\geq 2$-dimensional (v3.0 joint SPX/VIX).

**v2.0 sequencing inside the codebase:** Gaussian-to-Gaussian sanity check (closed form) → martingale Sinkhorn on SPX → LightSB on SPX → Henry-Labordère martingale-projected reference dynamics for arbitrage consistency → DSBM only if needed.

### 2.4 Multi-marginal extension (v3.0)

For $M \geq 2$ marginals, two approaches:
- **Sequential IPF** on consecutive maturity pairs (Pavon–Tabak–Trigila 2021): loses jointness but cheap. **v2.0 uses this.**
- **Coupled multi-marginal IPF** (Chen–Georgiou–Pavon 2021, Noble et al. 2023): one bridge that simultaneously matches all marginals. More principled. **v3.0 uses this.** Acharya et al. (2026) show that per-maturity "Markovian stitching" can be strictly sub-optimal for SPX/VIX; their entropic mirror-descent scheme is the reference point for v3.0.

### 2.5 Deep hedging (v1.0 minimal, v2.0 polished)

Train a hedging policy $\pi_\phi: (t, S_t, \text{book state}_t) \mapsto \text{position}$ to minimize a convex risk measure of terminal P&L (Buehler et al. 2019):

$$\mathcal L(\phi) = \rho\!\left(-\Pi^{(n)}_T(\pi_\phi)\right), \qquad \rho = \text{CVaR}_\alpha.$$

Terminal P&L is a differentiable function of the positions along each simulated path, so the objective is trained by **pathwise backpropagation** through the P&L (as in Buehler et al.), not by REINFORCE. **Proportional transaction costs are part of the v1.0 spec**: without them the optimal hedge of a vanilla is essentially delta and there is nothing for the policy to learn. **v1.0:** hedging a vanilla call with the underlying under costs, on Heston-driven vs. Dupire-driven vs. neural-LSV-driven scenarios, evaluated cross-model (train on A, test on B) and on historical SPX windows. **v2.0:** exotics, fuller comparison.

---

## 3. Foundations inherited from Skoltech HWs

| Skoltech HW | What it taught | Where it lands |
|---|---|---|
| OptionPricing.ipynb: GBM, Itô, Fokker–Planck, BS | Foundational option pricing | Reference dynamics, Heston baseline |
| OptionPricing.ipynb: Heston + implied vol | Calibrating 2-factor stoch vol to AAPL | Direct parent of v1 baseline |
| HW3 P1–P2: numerical Itô, Probability Flow ODE | SDE-to-ODE bridge, score-implied dynamics | Core of v2 score-based SB engine |
| HW3 P3: Rectified flow | OT geodesics as self-improving generative model | Bridge-matching regression step in v2 SB solvers |
| HW4 P1–P4: MCMC, Fokker–Planck via Fourier | Bayesian sampling, distribution recovery | Posterior inference; sanity checks |
| HW4 P5: yfinance + Heston calibration + MC pricing | Complete real-data calibration pipeline | Exact pattern reused for v1 calibration loop |

The repo's story arc: *"these are the well-understood building blocks I'm standing on; here is what's new."*

---

## 4. Release roadmap

### v1.0 — SPX + Neural LSV + Basic Deep Hedging (5-6 months)

**Goal**: working end-to-end pipeline on real SPX option data, with two classical baselines (Heston, Dupire) and one learned model (neural LSV), evaluated identically on a clean benchmark.

**Critical scope discipline**: no Schrödinger bridges in v1. No joint SPX/VIX. No crypto. No productization. **No exceptions.** Adding these to v1 is what kills v1.

#### Phase 1 — Data + classical baselines (~3-4 weeks) — ✅ **COMPLETE**

- ✅ ETL: pull SPX option chains via yfinance (free) and SPX time series. Cache with parquet.
- ✅ Cleaning: no-arbitrage filters (butterfly + calendar), liquidity filters, wing extrapolation (SVI fit), Breeden–Litzenberger marginal extraction with kernel smoothing.
- ✅ Heston calibration via MCMC, ported from HW4 P5; this is the v0 model.
- ✅ Evaluation harness: surface RMSE, marginal Wasserstein distance, MC option pricing error.

**Deliverables**:
- ✅ `notebooks/00_data_and_surface.ipynb` — SPX IV surface plot, recovered marginals at 8 maturities
- ✅ `notebooks/01_heston_mcmc.ipynb` — Heston calibrated to a recent SPX snapshot, with all benchmark numbers
- ✅ `src/data/`, `src/eval/` (the plan said `lib/`; see the §5 amendment)

**Phase 1 exit criterion**: I can show a plot of the SPX implied vol surface, the recovered marginals, and a table of Heston calibration metrics. **Done.** Benchmark row `heston-nuts` on the 2026-06-16 snapshot: in-sample IV RMSE 0.84 vol points, **held-out 2.66**, SVI floor 0.52, Feller violated.

**Three things Phase 1 got wrong on the first pass, and what they cost.** Worth remembering, because
each was silent — the code ran and produced plausible numbers either way:

1. **The BL grid must stop at the last traded strike.** Beyond it, BL reads the density off SVI's
   *extrapolated* wing, which invents tail mass; since `E[S_T] = ∫K μ(K) dK` weights by K, that fake
   mass dominates. A grid to 1.5F or 3F gave a **+23–27%** forward-recovery error at mid maturities;
   the traded-range grid gives **<1%**.
2. **The Fourier pricer's `u_max` is strained by SHORT maturities, not long ones** (the CF decays
   like `exp(-c·v·T·u²)`). The inherited `(u_max=200, n_quad=128)` carried a **0.63 absolute price
   error at T=0.017** — inside the calibration objective. Now `(800, 256)`.
3. **Held-out scoring changes the story.** In-sample IV RMSE was 0.84 vol points; on the held-out
   wing it is **2.66**. Both are real; only the second is a forecast.

**Two things the Phase 2 review found (2026-09), recorded before they cost anything:**

4. **The parity-implied rate spreads 3.8 percentage points across expiries**, while the code uses one
   flat `r` and ignores dividends. Phase 2 keeps the flat `r` so all three benchmark rows share one
   forward convention; per-expiry forwards from put-call parity are the first v2.0 prerequisite
   (`Paths` gains an optional per-time `forward`, default `s0·e^{rt}`).
5. **W1 against the BL marginals is a truncated-density metric** (`tail_mass` up to 8.6%): even Dupire
   will not score zero. `iv_rmse_test` and the forward-smile column are the discriminating numbers.

#### Phase 2 — Dupire baseline + neural LSV on SPX (~4-6 weeks)

- Harness additions (model-agnostic, `src/eval`): shared maturity-exact `time_grid`; `forward_start_iv` (the dynamics metric, §8); optional `mc_price_*`/`feller` and a new `fwd_iv_atm` column in the benchmark row; heatmap / training-curve / forward-smile figures.
- **Dupire local vol** from the SVI surface (§2.1), log-Euler Monte Carlo → `Paths`; benchmark row `dupire-lv`. This is the marginal-matching ceiling.
- **Neural LSV** (§2.2): torch Euler simulator returning `Paths`; leverage net; CIR factor initialised from the NUTS posterior mean; vega-weighted price loss on the train wing split; ~500 Adam steps, minutes on the M4. Benchmark row `neural-lsv`.
- Diagnostics: smile fit on train/test, $L_\theta$ heatmap, martingale check, forward smiles of all three models on one figure, training curve.
- Tests under the 100% gate: seam test (with $L \equiv 1$ the torch Euler equals the numpy Heston Euler to round-off), hypothesis property test that $\mathbb E[S_T]$ equals the forward within stderr for random parameters and a random non-trivial leverage net, torch pricer equals `mc_call_prices` on the same tensors.

**Deliverables**:
- `notebooks/02_local_vol.ipynb` — Dupire surface, simulated marginals, benchmark row
- `notebooks/03_neural_lsv_spx.ipynb` — neural LSV trained on the real SPX snapshot, benchmark row, forward-smile comparison
- `src/algorithms/local_vol/`, `src/algorithms/neural_lsv/`

**Phase 2 exit criterion**: the benchmark table has rows `heston-nuts`, `dupire-lv`, `neural-lsv` on the same snapshot; neural LSV is at or below Heston on `iv_rmse_test` (2.66) and near the SVI floor in-sample; martingale $|z| < 3$ for all rows; one figure shows the three models' forward smiles differ. (Matching the marginals alone is not a criterion — Dupire does that by construction.)

#### Phase 3 — Deep hedging (~3-4 weeks)

- Implement Buehler et al. 2019 with a small MLP policy in torch (~150 lines; the reference `deephedging` repo is TensorFlow), trained by pathwise backprop of CVaR$_{5\%}$.
- Proportional transaction costs from day one.
- Train on neural-LSV, Dupire and Heston scenarios; evaluate **cross-model** (every train/test pair) and on held-out historical SPX windows (10 years of daily data give ~120 non-overlapping one-month windows; sliding windows overlap heavily — say so).
- Report mean P&L, std, CVaR$_{5\%}$, turnover, cost paid.

**Deliverables**:
- `notebooks/04_deep_hedging.ipynb` — policy trained on each scenario source, P&L distributions plotted
- `src/algorithms/hedging/`

**Phase 3 exit criterion**: I can produce a single chart showing P&L distributions for delta-only, Heston-policy, Dupire-policy, neural-LSV-policy hedging under costs, with CVaR numbers, and the cross-model table.

#### v1.0 ship (~2 weeks)

- **Blog post #1**: "Calibrating a neural local-stochastic-volatility model to the SPX surface — and why the vanillas don't tell you the dynamics." Aimed at quant Twitter and r/quantfinance.
- README polish, ensure repo runs cleanly from clean clone.
- Public announcement.

---

### v2.0 — Schrödinger Bridge Upgrade (2-3 months after v1.0)

**Goal**: replace "a calibrated LSV" with "the entropically selected LSV", add Henry-Labordère martingale projection, polish the deep hedging engine.

Prerequisite: per-expiry forwards from put-call parity (Phase 1 lesson 4).

- Gaussian-to-Gaussian SB sanity check (closed form).
- **Martingale Sinkhorn** on the strike grid between consecutive BL marginals (De March & Henry-Labordère 2019). Exact, no neural nets.
- **Light Schrödinger Bridge** (Korotin et al. 2024) on SPX; compare to the Sinkhorn solution.
- Henry-Labordère martingale-projected reference dynamics.
- DSBM / IPF with score networks **only if** the grid and LightSB solutions prove insufficient.
- Deep hedging extended to exotics (barriers, autocalls).
- Robustness analysis: $\pm 5\%$ surface bumps, sensitivity reporting.

**Deliverables**:
- `notebooks/05_sb_sanity.ipynb`, `notebooks/06_martingale_sinkhorn_spx.ipynb`, `notebooks/07_deep_hedging_exotics.ipynb`
- `src/algorithms/bridges/`

**v2.0 ship**:
- **Blog post #2**: "Schrödinger bridges as market generators."
- Workshop paper draft: target ICAIF or NeurIPS ML4Finance.

---

### v3.0 — Joint SPX/VIX Calibration (3 months after v2.0)

**Goal**: joint SPX/VIX calibration on a representative date with the SB-trained neural SDE.

**What changed since v0.3.** Joint calibration is no longer "the open problem": Guyon & Mustapha (2023) calibrated a neural SDE jointly to SPX smiles, VIX futures and VIX smiles within bid-ask (and found the learned vol factor is path-dependent and mean-reverting); Guyon & Lekeufack (2023, 4-factor PDV) and Abi Jaber, Illand & Li (2022, quintic OU) do it parametrically; Cuchiero et al. (2025) with signature models; Acharya et al. (2026) with multi-maturity entropic selection. None of the neural results has public code.

**Contribution options, pick one at v3.0 kickoff:**
1. **Open-source reproduction** of neural joint calibration on the shared harness — valuable on its own.
2. **Selection**: among the many jointly-calibrated models, the one closest in KL to a reference (the SB angle; continuous with v2.0).
3. **Evaluation-driven comparison** of PDV vs. quintic-OU vs. neural on the harness, with hedging P&L as the tie-breaker.

- Pull VIX option chain; compute Breeden–Litzenberger VIX marginals.
- Augment loss with VIX-derived constraints (Carr–Lee 2009; Guyon 2022).
- Coupled multi-marginal IPF (now jointly matching all marginals).
- Benchmark against Guyon–Lekeufack PDV and rough Bergomi.

**Deliverables**:
- `notebooks/08_joint_spxvix.ipynb`
- Results section in `reports/benchmark.md`

**v3.0 outcomes** (acceptable in decreasing order of awesomeness):
1. Joint calibration succeeds cleanly with a clear selection story. Publishable as full research paper.
2. Partial success — captures some joint structure but not all. Publishable with honest discussion of failure modes; still a contribution.
3. Doesn't work. Document precisely what didn't work and why. This is itself a contribution. Fall back to v2.0 as the headline deliverable.

---

### v4.0 — Crypto Extension + Productization (2-3 months after v3.0)

**Goal**: the universally accessible, regulation-light, Dubai-relevant slice. Plus packaging.

- Deribit BTC/ETH option-chain ETL.
- Rerun the full v1-v2 pipeline on BTC.
- Adapt for perpetual funding rates (crypto's analogue of dividends).
- Streamlit demo: upload option chain → calibrate → return fan chart + exotic prices + hedge P&L.
- Package: `pyproject.toml`, GitHub Actions CI, `mkdocs-material` docs, PyPI release `quantflow-0.1.0`.

**Deliverables**:
- `notebooks/09_btc_calibration.ipynb`, `apps/streamlit_demo.py`, `src/data/deribit.py`
- Full package layout with PyPI release
- **Blog post #3**: "QuantFlow: open-source generative market models."
- Submit workshop paper.

---

## 5. Repo structure (evolves with releases)

### v1.0 structure — AMENDED (Phase 1)

> **This section is superseded by what was actually built.** The plan called for flat modules under
> `lib/`. Two things killed that: `.gitignore` ignores `lib/` (the standard Python-template rule for
> build output), and the code grew natural sub-groupings well before v4. It lives in `src/` as a
> package tree. `CLAUDE.md` documents the real layout.

```
quantflow/
├── PROJECT_PLAN.md                       # this document
├── pyproject.toml                        # uv-managed (NOT requirements.txt)
├── notebooks/
│   ├── 00_data_and_surface.ipynb         # Phase 1
│   ├── 01_heston_mcmc.ipynb              # Phase 1
│   ├── 02_local_vol.ipynb                # Phase 2
│   ├── 03_neural_lsv_spx.ipynb           # Phase 2
│   └── 04_deep_hedging.ipynb             # Phase 3
├── src/
│   ├── data/                             # SPX ETL
│   ├── algorithms/
│   │   ├── black_scholes.py
│   │   ├── preprocessing/                # no-arb filters, SVI, BL
│   │   ├── heston/                       # Heston + MCMC + MC
│   │   ├── local_vol/                    # Phase 2: Dupire surface + log-Euler MC
│   │   ├── neural_lsv/                   # Phase 2: torch simulator, leverage net, loss, training
│   │   └── hedging/                      # Phase 3
│   └── eval/                             # all metrics
├── data/                                 # gitignored
└── reports/
    └── benchmark.md                      # numbers table
```

**The one rule that matters:** `src/eval` never imports a *model or calibrator* from
`src/algorithms`. §8's "every method is evaluated identically" only holds if the harness cannot see
which model it is scoring — so models depend on the harness's `Paths` contract, never the reverse.
Shared data carriers (`Marginal`, `SVIParams`) and the pure Black–Scholes helpers are the allowed
exceptions; they carry no model.

### v4.0 structure (mature)

Refactor to the full `src/quantflow/` package layout at v4.0, when the abstractions are actually known. Not before.

---

## 6. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Validated. C++ is wrong tool for this project. |
| Arrays | numpy + jax | jax for the Fourier pricer and NUTS; numpy for the classical MC simulators |
| Deep learning | PyTorch (primary) + jax (samplers) | torch for the neural LSV and deep hedging; numpyro is jax-native |
| Neural SDE | **hand-written Euler in torch** | *Amended (Phase 2).* torchsde is inactive (its maintainers point to Diffrax) and the stochastic adjoint is a memory optimisation with worse gradients; for 1D and ~260 steps plain backprop through the solver is exact and faster. torchsde removed from the dependencies. |
| Probabilistic | numpyro (NUTS) | Heston MCMC baseline |
| Options data | yfinance (free), Deribit API (v4) | Free sources only for v1; paid CBOE if scope grows |
| DataFrames | **pandas** | *Amended (Phase 1).* Originally "polars primary". The option-chain work is small-N, wide-column and sits directly on yfinance/pyarrow, both of which hand back pandas — so polars never earned its place. polars and altair removed from the dependencies (Phase 2). |
| Plotting | matplotlib | static for paper; interactive plotting deferred to the v4 demo |
| App | Streamlit (v4 only) | Skip until v4 |
| Tests | pytest, hypothesis | property-based tests on SDE samplers; 100% coverage is a hard gate |
| CI | GitHub Actions | Free for public repos |
| Compute | M4 Mac; cloud GPU not before v3 | See Section 7 |

**Optional polish for v4.0**: write the Monte Carlo inner loop for exotic pricing in Rust or C++ with PyO3/pybind11 bindings, benchmark vs. pure Python. Demonstrates systems skills alongside math without distracting from the core project. Strictly optional and only after v3.0 ships.

---

## 7. Compute strategy

**Total expected compute spend across all 4 releases: $100-400.**

- **Local M4 Mac**: data loading, cleaning, plotting, MCMC, all v1.0 training (the neural LSV is 1D, 8 maturities, ~500 quotes, ~260 steps — about two minutes on the CPU in float64), all notebook development, debugging. **No cloud GPU is needed before v3.0**; do not let "I need to set up the cloud box" become the blocker.
- **Cloud GPU (Vast.ai or Runpod, RTX 3090-class, ~$0.30/hour)**: training runs longer than ~30 minutes (v3 joint calibration, v4 crypto reruns). Spin up → sync code → train → sync results → shut down.
- **Do not buy a GPU for this project.** The compute pattern is bursty; rental is correct.

Workflow:
1. Develop locally on M4.
2. When you have a training run >30 min, push code to GitHub, spin up cloud instance, pull code, run, save artifacts to S3 or scp back.
3. Don't keep cloud instances running idle. Stop them as soon as the run finishes.

**Data accumulation (start now).** There is one option-chain snapshot (2026-06-16). v2.0's robustness bumps and Phase 3's evaluation want several. A daily pull during market hours into dated parquets costs nothing and compounds.

---

## 8. Evaluation protocol

The cardinal rule: **every method is evaluated identically.** The benchmark table itself is a contribution.

**Calibration accuracy.** Implied vol RMSE on held-out wing of strikes (the headline). Marginal Wasserstein vs. Breeden–Litzenberger (reported with `tail_mass`; truncated-density caveat). SVI fit residuals (the noise floor).

**Dynamics (v1+).** Vanillas do not identify dynamics (§1.3), so the table carries at least one number they do not pin: the **forward-start ATM implied vol** between two traded maturities, computed model-agnostically from `Paths` as $\mathbb E[(S_{T_2}/S_{T_1} - 1)^+]$ inverted through Black–Scholes, with its MC standard error. Local vol flattens the forward smile; stochastic vol does not; this is where the rows differ.

**Distributional fidelity.** CRPS of simulated $S_T$ vs. realized. PIT histograms. Coverage of 50/80/95 prediction intervals.

**No-arbitrage.** Martingale residual $\mathbb E[S_T] - S_0 e^{rT}$, in standard-error units; never corrected on the paths. Butterfly and calendar arbitrage tests on simulated surfaces.

**Exotic pricing (v2+).** 5 standard exotics: barrier, autocall, variance swap, forward-start, cliquet.

**Deep hedging P&L.** Mean, std, CVaR$_{5\%}$, turnover, cost paid; cross-model train/test matrix.

**Robustness (v2+).** $\pm 5\%$ surface bump; report price/hedge sensitivity.

All results land in `reports/benchmark.md` as a single growing table. Every Monte-Carlo number carries its standard error.

---

## 9. Realistic milestones for v1.0

Pace targets, not deadlines. Hitting them within ±50% is fine; "consistency" is not the goal.

| Approx. month | Milestone |
|---|---|
| 1 | Repo scaffold, README, yfinance ETL, raw SPX vol surface plot. No-arb filters and BL marginal extraction implemented. ✅ |
| 2 | Heston MCMC ported from HW4 P5; first surface RMSE number; benchmark table v0. ✅ |
| 3 | Dupire local-vol baseline and harness additions (forward-smile metric); second benchmark row. |
| 4 | Neural LSV wired into the torch simulator; price-loss training on SPX; third benchmark row. |
| 5 | Deep hedging engine with costs; CVaR comparison across scenario sources. |
| 5-6 | Polish, blog post #1, public announcement. **v1.0 ships.** |

**Reality check**: this is part-time work alongside a CTO role, with bursty effort patterns. 6 months is the realistic target; 5 is the stretch; 9 is acceptable. If month 6 arrives and v1.0 isn't shipped, the question is not "work harder" but "what's blocking and how do I unblock."

**Hard re-evaluation gates:**
- **End of month 2**: Phase 1 complete? ✅
- **End of month 4**: Phase 2 complete? If no, consider shipping a "Phase 1 + Heston + Dupire + first neural-LSV results" partial v1.0 rather than waiting indefinitely.
- **End of month 6**: v1.0 shipped? If no, ship whatever exists as v0.5 with honest documentation of what's missing, then resume.

---

## 10. Career strategy

**Single primary value: portfolio piece for quant interviews.** Everything else is upside.

**Target firms, ranked by realistic probability:**

1. **Crypto market makers in Dubai** (Wintermute, GSR, Cumberland, Galaxy, Auros): aligned with relocation plan, value the crypto extension, sponsor visas, less Russian-passport friction. **Highest probability target.**

2. **Tier 2 quant firms and bank exotics desks** (JPM exotics, GS strats, Citi, Millennium platform pods, Balyasny, second-tier funds): value the SPX work and joint calibration angle. Real options once v2.0 ships.

3. **Tier 1 quant firms** (Citadel, Two Sigma, Jane Street, DE Shaw, HRT, Optiver, Jump): stretch targets. Realistic only with v3.0 shipped + workshop paper + senior endorsements. Treat as upside, not base case.

4. **ML research labs with finance interest** (some teams at Meta FAIR, Google Research): possible but narrow. The "diffusion for finance" angle has a credible research story.

**The artifact-to-interview mapping:**
- v1.0 → I can credibly apply to Tier 2 and crypto firms with real portfolio backing.
- v2.0 → Tier 2 becomes higher probability; Tier 1 becomes plausible.
- v3.0 → If joint calibration works, Tier 1 becomes real. If it doesn't, v2.0 + paper is still strong.
- v4.0 → Productization angle helps with all tiers; Streamlit demo is very interview-friendly.

**Publication path:**
- Workshop paper at NeurIPS ML4Finance or ICAIF after v2.0. Submission deadlines around July–August.
- Longer-form journal (SIAM JFM, Quantitative Finance) only if v3.0 produces something publishable.

**Senior-researcher outreach** (do this at v1.0 ship, not v3.0):
- Email Buehler, Teichmann, Cuchiero, or Guyon with a specific question or draft work. The LightSB connection (Korotin, Burnaev — Skoltech) is a natural in-house conversation at v2.0.
- A senior endorsement at month 6-9 of the project changes hiring outcomes more than any additional code.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Project doesn't ship at all due to discipline pattern** | Hard scope discipline on v1.0. Re-evaluation gates at months 2, 4, 6. Accept v0.5 partial ship rather than indefinite delay. |
| **Neural LSV training is noisy or unstable** | Starts exactly at Heston (L≡1), regulariser shrinks to Heston, price loss with vega weights, antithetic paths, fixed evaluation grid. Dupire row is the fallback deliverable. |
| **SB IPF is numerically temperamental** | Deferred to v2.0, and v2.0 starts with exact grid Sinkhorn and LightSB, not IPF. Gaussian sanity check before real data. DSBM only if needed. |
| **Real option data quality issues** | Aggressive liquidity filtering. SPX only for v1 (deepest book). Free yfinance data; consider paid CBOE only if data quality blocks progress. Start daily snapshot accumulation now. |
| **Joint SPX/VIX doesn't work** | It has been done; the risk is *contribution*, not feasibility. v3.0 picks one of three contribution framings at kickoff; "informative negative result" remains acceptable. |
| **Deep hedging finicky to train** | Pathwise gradients (not REINFORCE), transaction costs, start with delta-hedging-a-call before exotics. |
| **Scope creep** | Each release has explicit deliverables and a hard ship gate. v1.0 explicitly does NOT include SB, joint calibration, or crypto. |
| **Compute cost** | No cloud before v3; then $0.30/hr cloud GPU, bursty usage, total project budget $100-400. Bounded. |
| **You publish and nobody finds it** | Three blog posts at v1.0, v2.0, v4.0 ship. Twitter/X threads. Email a draft to one senior person by v2.0 ship. Submit to ICAIF/NeurIPS workshop. |
| **Burnout / loss of interest in 9 months** | The release structure means partial completion still has value. Stopping at v1.0 is fine. Stopping at v2.0 is great. Stopping at v3.0 is a publishable contribution. |

---

## 12. Naming, hosting, licensing

- **Name**: `quantflow`. Available on PyPI (check before PyPI release in v4.0).
- **Repo**: `github.com/<your-handle>/quantflow`, public from day one.
- **License**: Apache-2.0 for code; CC-BY-SA 4.0 for notebooks and blog posts.
- **Branding**: minimal README, one fan-chart figure, one benchmark table. Math speaks.

---

## 13. Reading list, split by release

Read by release. Do not read ahead of the release you are building.

### For v1.0 (read first)

**Local volatility, Gyöngy, LSV (the v1 engine):**
- Dupire, B. (1994). *Pricing with a Smile.* Risk.
- Gyöngy, I. (1986). *Mimicking the one-dimensional marginal distributions of processes having an Itô differential.* PTRF.
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley. (Ch. 1 for Dupire in total-variance form.)
- Guyon, J., Henry-Labordère, P. (2012). *Being particular about calibration.* Risk. (The particle method.)

**Neural SDE calibration:**
- Cuchiero, C., Khosrawi, W., Teichmann, J. (2020). *A GAN Approach to Calibration of Local Stochastic Volatility Models.* Risks. (Neural leverage function.)
- Gierjatowicz, P. et al. (2020). *Robust Pricing and Hedging via Neural SDEs.* arXiv:2007.04154. (Price loss, control variates.)
- Kidger, P. et al. (2021). *Neural SDEs as Infinite-Dimensional GANs.* ICML.

**Deep hedging:**
- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging.* Quantitative Finance.

### For v2.0 (read after v1.0 ships)

**Schrödinger bridges:**
- Léonard, C. (2014). *Survey of the Schrödinger problem and OT connections.*
- De March, H., Henry-Labordère, P. (2019). *Building arbitrage-free implied volatility: Sinkhorn's algorithm and variants.* SSRN.
- Henry-Labordère, P. (2019). *From (martingale) Schrödinger bridges to a new class of stochastic volatility models.* arXiv:1904.04554.
- Korotin, A., Gushchin, N., Burnaev, E. (2024). *Light Schrödinger Bridge.* ICLR.
- Gushchin, N., Kholkin, S., Burnaev, E., Korotin, A. (2024). *Light and Optimal Schrödinger Bridge Matching.* ICML.
- De Bortoli, V. et al. (2021). *Diffusion Schrödinger Bridge.* NeurIPS.
- Shi, Y. et al. (2023). *Diffusion Schrödinger Bridge Matching.* NeurIPS.
- Vargas, F. et al. (2021). *Solving SB via Maximum Likelihood.* Entropy.

**Flow matching (the regression step inside bridge matching):**
- Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling.* ICLR.
- Liu, X., Gong, C., Liu, Q. (2022). *Flow Straight and Fast: Rectified Flow.* ICLR. *(HW3 P3 paper)*
- Tong, A. et al. (2023). *Improving and Generalizing Flow-Based Generative Models with Minibatch OT.*
- Albergo, M., Boffi, N., Vanden-Eijnden, E. (2023). *Stochastic Interpolants: A Unifying Framework for Flows and Diffusions.*

**Arbitrage-free:**
- Cohen, S., Reisinger, C., Wang, S. (2023). *Arbitrage-Free Neural-SDE Market Models.*

### For v3.0 (read during v2.0)

**Joint SPX/VIX:**
- Guyon, J., Mustapha, S. (2023). *Neural Joint S&P 500/VIX Smile Calibration.* Risk.
- Guyon, J., Lekeufack, J. (2023). *Volatility is (mostly) path-dependent.* Quantitative Finance.
- Abi Jaber, E., Illand, C., Li, S. (2022). *Joint SPX–VIX calibration with Gaussian polynomial volatility models.*
- Cuchiero, C., Gazzani, G., Möller, J., Svaluto-Ferro, S. (2025). *Joint calibration to SPX and VIX options with signature-based models.* Mathematical Finance.
- Acharya, et al. (2026). *Global Multi-Maturity SPX–VIX Calibration Beyond Markovian Stitching.* arXiv:2609.04087.
- Guyon, J. (2022). *The smile of stochastic volatility.* Risk.
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under Rough Volatility.*

### For v4.0 / a P-measure market simulator (read if time permits)

These train SDEs on *observed paths*, not on option prices — relevant to a scenario generator for hedging, not to calibration:
- Bartosh, G., Vetrov, D., Naesseth, C. (2025). *SDE Matching: Scalable and Simulation-Free Training of Latent SDEs.* ICML.
- Zhang, J. et al. (2024). *Efficient Training of Neural SDEs by Matching Finite Dimensional Distributions.*
- Issa, Z., Horvath, B., Lemercier, M., Salvi, C. (2023). *Non-adversarial training of Neural SDEs with signature kernel scores.* NeurIPS.
- Holderrieth, P. et al. (2024). *Generator Matching.*
- Wang, X., Després, A., Dureau, M., Buet-Golfouse, F. (2026). *Amortizing the Calibration Triple: A Projection-Consistent Neural Operator for LSV.* arXiv:2608.01217. (Evidence the field is on the LSV/Gyöngy framing.)
- Wiese, M. et al. (2020). *Quant GANs.*; Ni, H. et al. (2021). *Sig-Wasserstein GANs.*; Liu, G.-H. et al. (2023). *I²SB.*; El Euch, O., Rosenbaum, M. (2019). *Rough Heston.*

---

## 14. What to do next, concretely: Phase 2

In this order, each step gated by `uv run pytest` (100% coverage) and `uv run ruff check src tests`:

1. **Harness additions** (`src/eval`): move the maturity-exact `time_grid` next to `Paths`; add `forward_start_iv`; make `mc_price_*` and `feller` optional in the benchmark row and add `fwd_iv_atm`; add heatmap / training-curve / forward-smile figures.
2. **Dupire surface** from the SVI slices, with the floor diagnostic and flat extrapolation; tests on flat and synthetic surfaces.
3. **Local-vol Monte Carlo** → `Paths`; `notebooks/02_local_vol.ipynb`; row `dupire-lv`.
4. **Neural LSV model and torch simulator** → `Paths`; seam test against the Heston Euler; hypothesis drift-pinning test.
5. **Loss and training loop**; torch pricer must equal `mc_call_prices` to round-off.
6. **`notebooks/03_neural_lsv_spx.ipynb`**; row `neural-lsv`; forward-smile figure with all three models. Update README and CLAUDE.md.

In parallel, at zero cost: start the daily option-chain snapshot pull (§7).

---

## 15. The single most important meta-rule

**Partial completion of this project is valuable.** That's the framing the original plan was missing.

- Stop at Phase 1 of v1.0: you have a clean SPX data pipeline and Heston baseline. Useful as a personal project showcase.
- Stop at v1.0: you have a serious portfolio piece, sufficient for Tier 2 / crypto firm interviews.
- Stop at v2.0: you have a workshop paper draft and a real research artifact.
- Stop at v3.0: you have a publishable contribution to an open problem.
- Reach v4.0: you have a productized library that demonstrates the full engineering + research + math + product stack.

**Every stopping point is fine.** What kills projects of this scope is the perfectionism that says "if it's not done, it's worthless." That's wrong. v1.0 is genuinely valuable on its own. Ship it as soon as it's worth shipping; don't wait for v2.0 to be done first.

---

*End of design doc v0.4. Phase 2 is specified; go build the Dupire row.*
