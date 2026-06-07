# QuantFlow — Project Plan v2

> **QuantFlow** — *Generative market models for option-chain calibration and deep hedging, on equity index (SPX/VIX) and crypto (BTC/ETH).*
>
> Author: Dmitriy (Skoltech, ex-MIPT)
> Status: Draft v0.3 — restructured for realistic execution
> License (intended): Apache-2.0 for the library, CC-BY-SA 4.0 for the notebooks and writeups.

---

## 0. TL;DR

Build the open-source library that does what every bank's exotics desk is privately rebuilding: a **generative model of arbitrage-consistent market dynamics** calibrated to option chains via flow matching and (later) Schrödinger bridges, with **deep hedging** as the downstream evaluation.

**Restructured for honest execution.** This project is split into four releases of increasing ambition. Each release is independently valuable — partial completion still produces a real artifact. The original v1 plan tried to deliver everything in 14 weeks; this v2 plan acknowledges that v1 alone is a 5-6 month part-time project, and structures the rest accordingly.

| Release | Scope | Realistic timeline | Career-relevant value |
|---|---|---|---|
| **v1.0** | SPX calibration via flow matching + Heston baseline + basic deep hedging + clean benchmark | 5-6 months | Strong portfolio piece. Sufficient for interviews at Tier 2 quant firms and crypto market makers. |
| **v2.0** | Schrödinger bridge upgrade + Henry-Labordère martingale projection | +3-4 months | Adds research depth. Sufficient for ICAIF / NeurIPS ML4Finance workshop submission. |
| **v3.0** | Joint SPX/VIX calibration (the open problem) | +3 months | Genuine research contribution if it works; informative negative result if it doesn't. |
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

$$dS_t \;=\; r\,S_t\,dt \;+\; S_t\,\sigma_\theta(t, S_t, V_t)\,dW_t, \qquad dV_t \;=\; b_\theta(t, S_t, V_t)\,dt + a_\theta(t, S_t, V_t)\,dB_t,$$

with $W,B$ correlated Brownian motions and $\sigma_\theta, b_\theta, a_\theta$ neural networks, such that:

1. **Marginal match**: simulated marginals $\mathrm{Law}_\mathbb{Q}(S_{T_i})$ equal $\mu_{T_i}$ for all $i$.
2. **No-arbitrage**: discounted $S_t$ is a martingale under $\mathbb Q$; the drift is structurally pinned at $r$.
3. **Joint surface match** (v3 only): simulated VIX$_t$ trajectories give VIX options prices that match the VIX surface simultaneously.
4. **Smoothness / regularity**: implied dynamics not pathological (no exploding variance), reproduce stylised facts (vol clustering, leverage effect, fat tails).

### 1.3 Why this is hard

Conditions (1) and (2) together — finding any martingale with prescribed marginals — has a unique solution only up to dynamics; this is the **multi-marginal martingale optimal-transport problem**. Classical SDE classes (Heston, SABR, rough Bergomi) parametrize a small slice of the solution space and so generically miss either some marginals or the dependence structure required for (3). Neural SDEs (Cuchiero et al. 2020, Gierjatowicz et al. 2020, Cohen et al. 2023) expand the function class but still leave the optimization problem under-specified.

**v1.0 approach: flow matching for tractability.** Use conditional flow matching (Lipman et al. 2023; Tong et al. 2023) to push $\mu_0$ to each $\mu_{T_i}$ along OT geodesics. Train a neural SDE whose simulated marginals match the flow-matched marginals. This is computationally cheap, well-conditioned, and gets us a working pipeline.

**v2.0 upgrade: Schrödinger bridge for principled selection.** Among all martingale measures with the prescribed marginals, pick the one of minimum relative entropy to a reference martingale measure. This is the unique entropic-OT solution: the most likely arbitrage-consistent market dynamics given the option chain.

The decision to ship v1 with flow matching only is deliberate. Flow matching is a defensible methodology in its own right; v1 is not "a worse version of v2," it's "a complete deterministic baseline that v2 then upgrades to stochastic."

### 1.4 Downstream evaluation

A market model is only as good as the decisions it informs.

- **Exotic pricing**: MC prices for autocallables, barriers, variance swaps, forward-start options, cliquets. Compare vs. street consensus where available and vs. Heston / rough Bergomi.
- **Deep hedging** (Buehler et al. 2019): train a hedging policy on scenarios from the market model; evaluate replication P&L on held-out historical paths. Lowest CVaR wins.
- **Robustness**: $\pm 5\%$ surface bump; report price/hedge sensitivity.

---

## 2. The mathematical engine

Different releases use different mathematical engines. This section describes both, in the order they appear in the codebase.

### 2.1 Flow matching (v1.0)

In the $\varepsilon \to 0$ limit of the Schrödinger problem, the bridge becomes an OT geodesic. **Conditional Flow Matching** (Lipman et al. 2023) trains a vector field $v_\theta(t, x)$ to satisfy

$$\mathcal L(\theta) = \mathbb E_{t \sim U[0,1],\ (x_0, x_T) \sim \pi}\Big[\,\big\|\,v_\theta(t, x_t) - (x_T - x_0)\,\big\|^2\,\Big], \qquad x_t = (1-t) x_0 + t x_T,$$

with $\pi$ an OT coupling (minibatch Sinkhorn). Solving the ODE $\dot x_t = v_\theta(t, x_t)$ pushes $\mu_0$ to $\mu_T$ deterministically. **Rectified flow** (Liu–Gong–Liu 2022, the HW3 P3 paper) is a self-improving variant.

For multi-marginal (the M traded maturities), v1.0 uses sequential flow matching: $\mu_0 \to \mu_{T_1}$, then $\mu_{T_1} \to \mu_{T_2}$, etc. Loses jointness but is computationally cheap and the right starting point.

### 2.2 Neural SDE realization (v1.0)

Given flow-matched targets, train a neural SDE

$$dS_t = r\,S_t\,dt + S_t\,\sigma_\theta(t, S_t, V_t)\,dW_t, \qquad dV_t = b_\theta(\cdot)\,dt + a_\theta(\cdot)\,dB_t$$

such that simulated marginals match. Direct adjoint training via `torchsde` (Kidger et al. 2021). Stochastic-Wasserstein loss vs. Breeden–Litzenberger marginals. Architectural drift pinning enforces the martingale constraint.

### 2.3 Schrödinger bridges (v2.0)

Given $\mu_0, \mu_T$ and reference path measure $\mathbb W$ on $C([0,T];\mathbb R^d)$, the **Schrödinger problem** is

$$\mathbb P^\star = \arg\min_{\mathbb P}\ \mathrm{KL}(\mathbb P \,\|\, \mathbb W) \quad \text{s.t. } \mathbb P_0 = \mu_0,\ \mathbb P_T = \mu_T.$$

Léonard (2014) is canonical. Solved computationally by **Sinkhorn on path space** (IPF), each half-step being a diffusion bridge regression implementable as score matching (De Bortoli et al. 2021; Shi et al. 2023 — IDSB makes IPF provably convergent without bias).

**v2.0 sequencing inside the codebase:**
- Sanity-check IPF on Gaussian-to-Gaussian SB (closed-form answer).
- Multi-marginal sequential IPF on SPX.
- Henry-Labordère martingale-projected reference dynamics for arbitrage consistency.

### 2.4 Multi-marginal extension (v3.0)

For $M \geq 2$ marginals, two approaches:
- **Sequential IPF** on consecutive maturity pairs (Pavon–Tabak–Trigila 2021): loses jointness but cheap. **v2.0 uses this.**
- **Coupled multi-marginal IPF** (Chen–Georgiou–Pavon 2021, Noble et al. 2023): one bridge that simultaneously matches all marginals. More principled. **v3.0 uses this.**

### 2.5 Deep hedging (v1.0 minimal, v2.0 polished)

Train a hedging policy $\pi_\phi: (t, S_t, \text{book state}_t) \mapsto \text{position}$ to minimize a convex risk measure of terminal P&L (Buehler et al. 2019):

$$\mathcal L(\phi) = \rho\!\left(-\Pi^{(n)}_T(\pi_\phi)\right), \qquad \rho = \text{CVaR}_\alpha.$$

REINFORCE training on simulated paths from the QuantFlow market model. **v1.0:** delta-hedging a vanilla call on Heston-driven vs. neural-SDE-driven scenarios. **v2.0:** exotics, fuller comparison.

---

## 3. Foundations inherited from Skoltech HWs

The activation energy for the early phases is near zero because every component has a direct ancestor in homework already done.

| Skoltech HW | What it taught | Where it lands |
|---|---|---|
| OptionPricing.ipynb: GBM, Itô, Fokker–Planck, BS | Foundational option pricing | Reference dynamics, Heston baseline |
| OptionPricing.ipynb: Heston + implied vol | Calibrating 2-factor stoch vol to AAPL | Direct parent of v1 baseline |
| HW3 P1–P2: numerical Itô, Probability Flow ODE | SDE-to-ODE bridge, score-implied dynamics | Core of v2 score-based SB engine |
| HW3 P3: Rectified flow | OT geodesics as self-improving generative model | First concrete flow matching realization |
| HW4 P1–P4: MCMC, Fokker–Planck via Fourier | Bayesian sampling, distribution recovery | Posterior inference; sanity checks |
| HW4 P5: yfinance + Heston calibration + MC pricing | Complete real-data calibration pipeline | Exact pattern reused for v1 calibration loop |

The first commit will be `git mv` of cleaned cells from these notebooks into `lib/baselines/` and `notebooks/00_inherited_baselines.ipynb`. The repo's story arc: *"these are the well-understood building blocks I'm standing on; here is what's new."*

---

## 4. Release roadmap

### v1.0 — SPX + Flow Matching + Basic Deep Hedging (5-6 months)

**Goal**: working end-to-end pipeline on real SPX option data, with one classical baseline (Heston) and one learned model (flow-matching neural SDE), evaluated identically on a clean benchmark.

**Critical scope discipline**: no Schrödinger bridges in v1. No joint SPX/VIX. No crypto. No productization. **No exceptions.** Adding these to v1 is what kills v1.

#### Phase 1 — Data + classical baselines (~3-4 weeks)

- ETL: pull SPX option chains via yfinance (free) and SPX time series. Cache with parquet.
- Cleaning: no-arbitrage filters (butterfly + calendar), liquidity filters, wing extrapolation (SVI fit), Breeden–Litzenberger marginal extraction with kernel smoothing.
- Heston calibration via MCMC, ported from HW4 P5; this is the v0 model.
- Evaluation harness: surface RMSE, marginal Wasserstein distance, MC option pricing error.

**Deliverables**: 
- `notebooks/00_data_and_baseline.ipynb` — SPX surface plot, recovered marginals at 5 maturities
- `notebooks/01_heston_mcmc.ipynb` — Heston calibrated to a recent SPX snapshot, with all benchmark numbers
- `lib/data/`, `lib/eval/`

**Phase 1 exit criterion**: I can show a plot of the SPX implied vol surface, the recovered marginals, and a table of Heston calibration metrics. Done.

#### Phase 2 — Flow matching neural SDE on SPX (~4-6 weeks)

- Implement the neural SDE class in `torchsde` with architecturally pinned drift.
- Conditional flow matching objective on synthetic samples from $\mu_{T_i}$ (deterministic warm-start).
- Adjoint training to minimize multi-marginal Wasserstein loss vs. Breeden–Litzenberger marginals.
- Diagnostics: density comparison plots, martingale check, variance term-structure, surface reconstruction.

**Deliverables**:
- `notebooks/02_flow_matching_warmup.ipynb` — rectified flow on 2D synthetic, reproducing HW3 P3 cleanly
- `notebooks/03_neural_sde_spx.ipynb` — full flow-matching-warmed neural SDE on real SPX data, benchmark numbers
- `lib/sde/neural.py`, `lib/training/flow_matching.py`

**Phase 2 exit criterion**: I have a trained neural SDE whose simulated marginals match the SPX marginals within target RMSE, with a benchmark table comparing it to Heston.

#### Phase 3 — Deep hedging (~3-4 weeks)

- Implement Buehler et al. 2019 with a small MLP policy.
- Train on QuantFlow scenarios and on Heston scenarios for comparison.
- Evaluate on held-out historical SPX paths.
- Report mean P&L, std, CVaR$_{5\%}$, turnover.

**Deliverables**:
- `notebooks/04_deep_hedging.ipynb` — policy trained on both scenario sources, P&L distributions plotted
- `lib/hedging/`

**Phase 3 exit criterion**: I can produce a single chart showing P&L distributions for delta-only, Heston-policy, QuantFlow-policy hedging, with CVaR numbers.

#### v1.0 ship (~2 weeks)

- **Blog post #1**: "Calibrating a neural SDE to the SPX option surface with flow matching." Aimed at quant Twitter and r/quantfinance.
- README polish, ensure repo runs cleanly from clean clone.
- Public announcement.

---

### v2.0 — Schrödinger Bridge Upgrade (3-4 months after v1.0)

**Goal**: replace the flow-matching engine with proper diffusion Schrödinger bridge IPF, add Henry-Labordère martingale projection, polish the deep hedging engine.

- Implement IPF with score networks regressed at each half-step.
- Sanity check on synthetic Gaussian-to-Gaussian SB (closed-form available).
- Multi-marginal **sequential** IPF on SPX (jointness deferred to v3).
- Henry-Labordère martingale-projected reference dynamics.
- Deep hedging extended to exotics (barriers, autocalls).
- Robustness analysis: $\pm 5\%$ surface bumps, sensitivity reporting.

**Deliverables**:
- `notebooks/05_dsb_sanity.ipynb`, `notebooks/06_sb_neural_sde_spx.ipynb`, `notebooks/07_deep_hedging_exotics.ipynb`
- `lib/bridges/`

**v2.0 ship**:
- **Blog post #2**: "Schrödinger bridges as market generators."
- Workshop paper draft: target ICAIF or NeurIPS ML4Finance.

---

### v3.0 — Joint SPX/VIX Calibration (3 months after v2.0)

**Goal**: solve the open joint SPX/VIX calibration problem on a representative date with the SB-trained neural SDE.

- Pull VIX option chain; compute Breeden–Litzenberger VIX marginals.
- Augment loss with VIX-derived constraints (Carr–Lee 2009; Guyon 2022).
- Coupled multi-marginal IPF (now jointly matching all marginals).
- Benchmark against Guyon path-dependent volatility and rough Bergomi.

**Deliverables**:
- `notebooks/08_joint_spxvix.ipynb`
- Results section in `reports/benchmark.md`

**v3.0 outcomes** (acceptable in decreasing order of awesomeness):
1. Joint calibration succeeds cleanly. Publishable as full research paper.
2. Partial success — captures some joint structure but not all. Publishable with honest discussion of failure modes; still a contribution.
3. Doesn't work. Document precisely what didn't work and why. This is itself a contribution; negative results in the joint calibration literature are valuable. Fall back to v2.0 as the headline deliverable.

---

### v4.0 — Crypto Extension + Productization (2-3 months after v3.0)

**Goal**: the universally accessible, regulation-light, Dubai-relevant slice. Plus packaging.

- Deribit BTC/ETH option-chain ETL.
- Rerun the full v1-v2 pipeline on BTC.
- Adapt for perpetual funding rates (crypto's analogue of dividends).
- Streamlit demo: upload option chain → calibrate → return fan chart + exotic prices + hedge P&L.
- Package: `pyproject.toml`, GitHub Actions CI, `mkdocs-material` docs, PyPI release `quantflow-0.1.0`.

**Deliverables**:
- `notebooks/09_btc_calibration.ipynb`, `apps/streamlit_demo.py`, `lib/data/deribit.py`
- Full package layout with PyPI release
- **Blog post #3**: "QuantFlow: open-source generative market models."
- Submit workshop paper.

---

## 5. Repo structure (evolves with releases)

### v1.0 structure (simple)

```
quantflow/
├── README.md
├── PROJECT_PLAN.md                       # this document
├── LICENSE                               # Apache-2.0
├── requirements.txt                      # pip install -r requirements.txt
├── notebooks/
│   ├── 00_data_and_baseline.ipynb        # Phase 1
│   ├── 01_heston_mcmc.ipynb              # Phase 1
│   ├── 02_flow_matching_warmup.ipynb     # Phase 2
│   ├── 03_neural_sde_spx.ipynb           # Phase 2
│   └── 04_deep_hedging.ipynb             # Phase 3
├── lib/                                  # not a package yet, just modules
│   ├── data.py                           # SPX ETL
│   ├── arbitrage.py                      # no-arb filters, SVI, BL extraction
│   ├── baselines.py                      # Heston, MCMC
│   ├── sde.py                            # neural SDE
│   ├── flow.py                           # flow matching
│   ├── hedging.py                        # deep hedging
│   └── eval.py                           # all metrics
├── data/                                 # gitignored
├── results/                              # gitignored, MC outputs etc.
└── reports/
    └── benchmark.md                      # numbers table
```

No premature packaging. No empty directories waiting to be filled. Just modules that exist and work.

### v4.0 structure (mature)

Refactor to the full `src/quantflow/` package layout at v4.0, when the abstractions are actually known. Not before.

---

## 6. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Validated. C++ is wrong tool for this project. |
| Arrays | numpy + jax | jax for vectorized SDE simulation and autodiff in samplers |
| Deep learning | PyTorch (primary) + jax (samplers) | torchsde is PyTorch-only; numpyro is jax-native |
| Neural SDE | `torchsde` (Kidger et al.) | Only mature library with adjoint backprop |
| Probabilistic | numpyro (NUTS) | Heston MCMC baseline |
| Options data | yfinance (free), Deribit API (v4) | Free sources only for v1; paid CBOE if scope grows |
| DataFrames | polars (primary), pandas (interop) | polars on time-series joins |
| Plotting | matplotlib + altair | static for paper, interactive for demo |
| App | Streamlit (v4 only) | Skip until v4 |
| Tests | pytest, hypothesis | property-based tests on SDE samplers |
| CI | GitHub Actions | Free for public repos |
| Compute | M4 Mac for local dev + Vast.ai / Runpod for training | See Section 7 |

**Optional polish for v4.0**: write the Monte Carlo inner loop for exotic pricing in Rust or C++ with PyO3/pybind11 bindings, benchmark vs. pure Python. Demonstrates systems skills alongside math without distracting from the core project. Strictly optional and only after v3.0 ships.

---

## 7. Compute strategy

**Total expected compute spend across all 4 releases: $100-400.**

- **Local M4 Mac**: data loading, cleaning, plotting, MCMC, small experiments, all notebook development, debugging.
- **Cloud GPU (Vast.ai or Runpod, RTX 3090-class, ~$0.30/hour)**: training runs longer than ~30 minutes. Spin up → sync code → train → sync results → shut down.
- **Do not buy a GPU for this project.** The compute pattern is bursty; rental is correct.

Workflow:
1. Develop locally on M4.
2. When you have a training run >30 min, push code to GitHub, spin up cloud instance, pull code, run, save artifacts to S3 or scp back.
3. Don't keep cloud instances running idle. Stop them as soon as the run finishes.

---

## 8. Evaluation protocol

The cardinal rule: **every method is evaluated identically.** The benchmark table itself is a contribution.

**Calibration accuracy.** Implied vol RMSE on held-out wing of strikes. Marginal Wasserstein vs. Breeden–Litzenberger. SVI fit residuals.

**Distributional fidelity.** CRPS of simulated $S_T$ vs. realized. PIT histograms. Coverage of 50/80/95 prediction intervals.

**No-arbitrage.** Martingale residual $\mathbb E[S_T] - S_0 e^{rT}$. Butterfly and calendar arbitrage tests on simulated surfaces.

**Exotic pricing (v2+).** 5 standard exotics: barrier, autocall, variance swap, forward-start, cliquet.

**Deep hedging P&L.** Mean, std, CVaR$_{5\%}$, turnover.

**Robustness (v2+).** $\pm 5\%$ surface bump; report price/hedge sensitivity.

All results land in `reports/benchmark.md` as a single growing table.

---

## 9. Realistic milestones for v1.0

Pace targets, not deadlines. Hitting them within ±50% is fine; "consistency" is not the goal.

| Approx. month | Milestone |
|---|---|
| 1 | Repo scaffold, README, yfinance ETL, raw SPX vol surface plot. No-arb filters and BL marginal extraction implemented. |
| 2 | Heston MCMC ported from HW4 P5; first surface RMSE number; benchmark table v0. |
| 3 | Flow-matching warmup (HW3 P3 generalized) on synthetic 2D marginals. |
| 4 | Neural SDE class wired into torchsde; adjoint training on SPX with FM warm-start; first calibration result. |
| 5 | Deep hedging engine; CVaR comparison vs. Heston scenarios. |
| 5-6 | Polish, blog post #1, public announcement. **v1.0 ships.** |

**Reality check**: this is part-time work alongside a CTO role, with bursty effort patterns. 6 months is the realistic target; 5 is the stretch; 9 is acceptable. If month 6 arrives and v1.0 isn't shipped, the question is not "work harder" but "what's blocking and how do I unblock."

**Hard re-evaluation gates:**
- **End of month 2**: Phase 1 complete? If no, the project's tempo is wrong and the scope of remaining phases needs adjustment.
- **End of month 4**: Phase 2 complete? If no, consider shipping a "Phase 1 + Heston + first FM results" partial v1.0 rather than waiting indefinitely.
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
- Email Buehler, Teichmann, Cuchiero, or Guyon with a specific question or draft work.
- A senior endorsement at month 6-9 of the project changes hiring outcomes more than any additional code.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Project doesn't ship at all due to discipline pattern** | Hard scope discipline on v1.0. Re-evaluation gates at months 2, 4, 6. Accept v0.5 partial ship rather than indefinite delay. |
| **SB IPF is numerically temperamental** | Deferred to v2.0. v1.0 doesn't depend on this. Gaussian sanity check before real data. Use IDSB (Shi et al. 2023) for convergence guarantees. |
| **Real option data quality issues** | Aggressive liquidity filtering. SPX only for v1 (deepest book). Free yfinance data; consider paid CBOE only if data quality blocks progress. |
| **Joint SPX/VIX doesn't work** | It's an open problem. v3.0 includes "informative negative result" as an acceptable outcome. Worst case, v2.0 is the headline deliverable. |
| **Deep hedging finicky to train** | Buehler's paper is well-documented; `deephedging` reference implementation exists. Start with toy delta-hedging-a-call before exotics. |
| **Scope creep** | Each release has explicit deliverables and a hard ship gate. v1.0 explicitly does NOT include SB, joint calibration, or crypto. |
| **Compute cost** | $0.30/hr cloud GPU, bursty usage, total project budget $100-400. Bounded. |
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

The original plan had one giant list of ~20 papers. This is a 12-month curriculum. Read by release.

### For v1.0 (read first)

**Flow matching (the v1 engine):**
- Lipman, Y. et al. (2023). *Flow Matching for Generative Modeling.* ICLR.
- Liu, X., Gong, C., Liu, Q. (2022). *Flow Straight and Fast: Rectified Flow.* ICLR. *(your HW3 P3 paper)*
- Tong, A. et al. (2023). *Improving and Generalizing Flow-Based Generative Models with Minibatch OT.*

**Neural SDE calibration:**
- Cuchiero, C., Khosrawi, W., Teichmann, J. (2020). *A GAN Approach to Calibration of Local Stochastic Volatility Models.* Risks.
- Gierjatowicz, P. et al. (2020). *Robust Pricing and Hedging via Neural SDEs.* arXiv:2007.04154.
- Kidger, P. et al. (2021). *Neural SDEs as Infinite-Dimensional GANs.* ICML.

**Deep hedging:**
- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging.* Quantitative Finance.

**Background reference:**
- Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley.

That's 8 papers + 1 book reference. Read these by end of Phase 1.

### For v2.0 (read after v1.0 ships)

**Schrödinger bridges:**
- Léonard, C. (2014). *Survey of the Schrödinger problem and OT connections.*
- De Bortoli, V. et al. (2021). *Diffusion Schrödinger Bridge.* NeurIPS.
- Shi, Y. et al. (2023). *Diffusion Schrödinger Bridge Matching.* NeurIPS.
- Vargas, F. et al. (2021). *Solving SB via Maximum Likelihood.* Entropy.

**Martingale SB:**
- Henry-Labordère, P. (2019). *From martingale SB to a new class of stoch vol models.*

**Arbitrage-free:**
- Cohen, S., Reisinger, C., Wang, S. (2023). *Arbitrage-Free Neural-SDE Market Models.*

### For v3.0 (read during v2.0)

**Joint SPX/VIX and rough vol:**
- Guyon, J. (2022). *The smile of stochastic volatility.* Risk.
- Guyon, J., Lekeufack, J. (2023). *Volatility is (mostly) path-dependent.* Quantitative Finance.
- Bayer, C., Friz, P., Gatheral, J. (2016). *Pricing under Rough Volatility.*

### For v4.0 / paper polish (read if time permits)

- Wiese, M. et al. (2020). *Quant GANs.*
- Ni, H. et al. (2021). *Sig-Wasserstein GANs.*
- Liu, G.-H. et al. (2023). *I²SB.*
- El Euch, O., Rosenbaum, M. (2019). *Rough Heston.*
- Lyons, T. et al. (2007). *Differential Equations Driven by Rough Paths.* Springer.

---

## 14. What to do next, concretely

In priority order, this week:

1. **Stop reading Schrödinger bridge material at the depth you've been reading it.** Léonard's survey at 30-40% depth is enough for v1.0. Defer the rest to month 6+.
2. **Set up the repo.** `git init` in the chosen directory. README skeleton. LICENSE. .gitignore. requirements.txt with the v1.0 dependencies. First commit.
3. **Day 1-2 work**: yfinance ETL pulling SPX option chain. Plot the surface. Commit.
4. **Day 3-5 work**: implement no-arb filters (Carr–Madan butterfly + calendar) and Breeden-Litzenberger marginal extraction. Plot the recovered marginals at 5 maturities. Commit.
5. **Day 6-7 work**: start the Heston MCMC port from HW4 P5. Get the sampler running on the new SPX data.

By end of week 1: repo exists, real data flowing, first real plots, foundation for Phase 1. This is more motivating than any amount of reading.


Lock this in before week 2.

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

*End of design doc v0.3. Ready to scaffold the repo.*
