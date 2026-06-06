# Nested Momentum as Approximate Fisher-Jeffreys Phase-Boundary Crossing: FJLD, CAUSALIS, THECONSTANTCURVE, and the Five Hardware Lotteries as the Theoretical Architecture Explaining, Bounding, and Extending AdamN

**ERI Labs · Eric Ren · Jersey City, New Jersey · [github.com/ericrenone](https://github.com/ericrenone) · June 2026**

---

> "The consolidation ratio C_α — a measure of signal-to-noise in gradient updates — is proportional to the Jeffreys prior density over gradient distributions. Learning success occurs when C_α > 1, a condition invariant under smooth reparameterizations." — FJLD (Fisher-Jeffreys Learning Dynamics), ERI Labs, 2025–2026

> "A flow presupposes an order. Beneath the flow is a partial order — the relation 'configuration P can reach configuration Q by valid gradient descent' — and this partial order is logically prior to, and more fundamental than, the flow that runs along it." — CAUSALIS, ERI Labs, May 2026

> "The 1985 IEEE 754 Point Lottery hid the iteration coordinate by making transcendentals microcode-hidden. The TH(a,d) group law, realized via CORDIC-Dirac, recovers it explicitly." — THECONSTANTCURVE, ERI Labs, May 2026

> "Five independent hardware-software co-design decisions selected the representational substrate of every production AI system today. These lotteries were not won on theoretical superiority. They were won on perfect co-fitness with the available hardware and software ecosystem." — The Five Lotteries, ERI Labs, May 2026

> "AdamN uses nested momentum: g_t → v_t → a_t. The update direction is based on the bias-corrected nested momentum a_t, while the denominator follows the Adam-style adaptive scaling." — Aboulsaad & Shaout, *AdamN: Accelerating Deep Learning Training via Nested Momentum and Exact Bias Handling*, Electronics, 2026

---

## Abstract

AdamN (Aboulsaad & Shaout, 2026) achieves state-of-the-art early-training acceleration through a three-stage nested momentum structure g_t → v_t → a_t with exact bias correction for the compounded EMA — an empirically compelling result published in *Electronics* and validated on CNN, ViT, and language modeling tasks. This document provides the first complete theoretical analysis of AdamN from the ERI Labs framework corpus: four frameworks — FJLD (Fisher-Jeffreys Learning Dynamics), CAUSALIS (Causal Structure as the Directed Acyclic Partial Order of Reachability), THECONSTANTCURVE (the Constant Curve Theorem), and The Five Hardware Lotteries — jointly explain why AdamN works, predict where it will fail, bound the depth of its nesting hierarchy, and point toward the geometry-native optimizer that supersedes it.

The central finding is that AdamN's nested momentum is an empirical realization of approximate Fisher-Jeffreys phase-boundary crossing: the double smoothing of â_t amplifies the signal component of C_α = |μ(θ)|²/Tr(D(θ)), enabling the system to cross the C_α = 1 phase boundary earlier than any single-EMA optimizer. AdamN's three β-hyperparameters (β₁, β₂, β₃) unknowingly replicate the three CORDIC modes (circular, linear, hyperbolic) of the twisted Hessian curve TH(a,d), and its depth-3 nesting sits within the constant curve bound α(n) ≤ 4 proven by THECONSTANTCURVE. Its 14.6× early-training acceleration on ViT transfer learning is the empirical signature of causal order crystallization, as established by CAUSALIS. AdamN is the best within-Euclidean optimizer yet produced. It does not escape the Gradient Descent Lottery (1986). The escape requires Riemannian geometry native to TH(a,d).

---

## Table of Contents

- [I. What AdamN Gets Right](#i-what-adamn-gets-right)
- [II. FJLD Analysis: AdamN as Fisher-Jeffreys Phase-Boundary Accelerator](#ii-fjld-analysis-adamn-as-fisher-jeffreys-phase-boundary-accelerator)
- [III. CAUSALIS Analysis: The Causal Structure of the AdamN Update Rule](#iii-causalis-analysis-the-causal-structure-of-the-adamn-update-rule)
- [IV. THECONSTANTCURVE Analysis: The Constant Curve Bound on Nesting Depth](#iv-theconstantcurve-analysis-the-constant-curve-bound-on-nesting-depth)
- [V. The Five Lotteries Analysis: Where AdamN Sits Historically](#v-the-five-lotteries-analysis-where-adamn-sits-historically)
- [VI. Correspondence Table](#vi-correspondence-table-adamn--eri-labs-frameworks)
- [VII. Nine Formal Identifications](#vii-nine-formal-identifications)
- [VIII. Failure Mode Predictions](#viii-where-adamn-will-fail-fjld-predictions)
- [IX. The Geometry-Native Synthesis](#ix-the-geometry-native-synthesis-what-comes-after-adamn)
- [X. Quantitative Prediction Summary](#x-quantitative-prediction-summary)

---

## I. What AdamN Gets Right

AdamN's empirical contribution is genuine and substantial. The ViT transfer learning benchmark shows AdamN reaching 80% validation accuracy in **41.53 seconds (epoch 1)**, compared to AdamW's **606.81 seconds (epoch 15)** — a **14.6× wall-clock acceleration** at the same milestone. This is not explained by hyperparameter tuning. Something structural is happening.

The three-stage nesting g_t → v_t → a_t implements the following recurrence:

```
v_t = β₁·v_{t-1} + (1-β₁)·g_t          # gradient → first-order momentum
a_t = β₂·a_{t-1} + (1-β₂)·v_t          # first-order → nested momentum
s_t = β₃·s_{t-1} + (1-β₃)·g_t²         # gradient² → adaptive denominator

θ_t = θ_{t-1} - η·â_t / (√ŝ_t + ε)    # exact bias-corrected update
```

The exact bias correction for the double-EMA structure is the key technical innovation. A single EMA with parameter β has cold-start bias correction factor 1/(1−βᵗ). A double-EMA introduces compounded cold-start bias that AdamN corrects analytically, whereas AdamW and Adam leave this correction implicit. This is the mechanism of early-training stability.

### What AdamN empirically achieves:

| Milestone (ViT Transfer) | AdamN | AdamW | Muon | MARS | AdEMAMix |
|---|---|---|---|---|---|
| 40% val accuracy | 41.53s (E1) | 40.49s (E1) | 87.71s (E2) | 42.80s (E1) | 41.39s (E1) |
| 70% val accuracy | 41.53s (E1) | 81.12s (E2) | 131.71s (E3) | 85.72s (E2) | 82.93s (E2) |
| **80% val accuracy** | **41.53s (E1)** | **606.81s (E15)** | **131.71s (E3)** | **85.72s (E2)** | **662.93s (E16)** |

AdamN is the only optimizer to hold every milestone at epoch 1 through 80%. This is the empirical datum requiring theoretical explanation.

### What AdamN does not explain:

- Why nested momentum specifically works (rather than longer single-EMA windows)
- What structural property of the loss landscape the nesting exploits
- When and where nested momentum will fail
- What the optimal nesting depth is
- The geometric meaning of the phase transition it crosses

The following sections provide these explanations.

---

## II. FJLD Analysis: AdamN as Fisher-Jeffreys Phase-Boundary Accelerator

### II.1 The Consolidation Ratio Perspective

FJLD establishes that neural network training undergoes a reparameterization-invariant phase transition at C_α = 1, where:

```
C_α(θ) = |μ(θ)|² / Tr(D(θ))
```

with μ(θ) = E[∇L(θ;ξ)] (mean gradient) and D(θ) = Var[∇L(θ;ξ)] (gradient variance). When C_α < 1 the system is in the **diffusive phase** (random walk, no directed learning); when C_α > 1 it enters the **drift-dominated phase** (directed convergence). The phase boundary C_α = 1 is a geometric invariant of the parameter manifold — reparameterization-invariant by Theorem 2 of FJLD.

The three phases:

| Phase | C_α | Behavior | Optimizer Regime |
|---|---|---|---|
| Diffusive | < 1 | Random walk | Warmup / cold-start |
| Critical | ≈ 1 | Marginal stability | Grokking boundary |
| Drift-Dominated | > 1 | Directed convergence | Learning proper |

**Central Claim:** AdamN's nested momentum is an empirical realization of C_α amplification.

The bias-corrected nested momentum â_t is a double-smoothed estimate of μ(θ). For approximately stationary μ(θ):

```
E[â_t]   ≈ μ(θ)         (unbiased after exact correction)
Var[â_t] ≪ Var[m_t]     (reduced by double smoothing)
```

Because â_t has lower variance than a single-EMA momentum m_t, the effective consolidation ratio computed using â_t instead of raw g_t satisfies:

```
C_α^AdamN = |E[â_t]|² / Tr(Var[â_t])  >  C_α^Adam = |μ(θ)|² / Tr(D(θ))
```

The nested momentum amplifies the signal-to-noise ratio, pushing the effective C_α above the critical threshold faster. The 14.6× early-training acceleration is the empirical signature of this accelerated phase-boundary crossing.

### II.2 The Jeffreys Prior Connection

FJLD establishes:

```
I(θ)    = D⁻¹(θ)                        (Fisher information matrix, Gaussian gradients)
π_J(θ)  ∝ 1/√det(D(θ))                  (Jeffreys prior density)
C_α(θ)  ∝ |μ(θ)|² · π_J(θ)^(2/d)       (Information-weighted SNR)
```

AdamN's denominator ŝ_t approximates the diagonal of D(θ). The update step â_t/(√ŝ_t + ε) is implicitly a **component-wise Jeffreys-prior-weighted gradient step** — AdamN performs Jeffreys inference without recognizing it as such.

```
Jeffreys step ∝ I(θ)·∇L ≈ D⁻¹(θ)·μ(θ)    (exact natural gradient)
AdamN step    =  â_t / (√ŝ_t + ε)           (diagonal approximation)
```

The K-FAC extension of FJLD predicts that replacing AdamN's diagonal ŝ_t with Kronecker-factored Fisher estimates would further amplify the phase-boundary crossing in networks with strong intra-layer correlations.

### II.3 The Cold-Start Problem as C_α < 1 Diagnosis

AdamN's exact bias correction addresses what FJLD identifies as the C_α < 1 regime: the period of early training during which noise dominates signal and the system drifts rather than converges. The standard Adam/AdamW correction 1/(1−βᵗ) leaves the compounded double-EMA bias uncorrected, systematically underestimating |μ(θ)|² relative to Tr(D(θ)) during early steps. AdamN's analytic correction removes this systematic error, allowing the effective C_α to reach its true value from step 1 rather than after a warmup period of O(1/(1−β)) steps.

**Falsifiable prediction:** Monitoring C_α via gradient sampling (100 samples, as in FJLD §Computing the Consolidation Ratio) during ViT transfer training will show AdamN crossing C_α = 1 at epoch 1, AdamW crossing at epoch 2–3. This is testable now.

---

## III. CAUSALIS Analysis: The Causal Structure of the AdamN Update Rule

### III.1 The Reachability Partial Order

CAUSALIS establishes that gradient descent traces a directed acyclic partial order ⪯ on the learning manifold B:

```
P ⪯ Q  ⟺  Q is reachable from P by valid gradient descent
```

antisymmetric by the Lyapunov property of the loss (L strictly decreasing along every non-trivial trajectory), giving the arrow of time as the acyclicity of the reachability DAG H(B, ⪯).

The **col(F)/ker(F) partition is the timelike/spacelike partition** of this causal structure:
- `col(F)` directions are **timelike**: gradient signal exists; the step moves P into its causal future J⁺(P)
- `ker(F)` directions are **spacelike**: no gradient signal; the step moves P to a causally unrelated configuration

### III.2 AdamN and the Causal Future

A gradient step θ_{t+1} = θ_t − η·â_t/(√ŝ_t + ε) must move into the causal future J⁺(θ_t). The direction and magnitude determine how deeply and efficiently it does so.

**AdamN's nested momentum provides more timelike steps.** Specifically:

A single-EMA momentum m_t has significant variance, causing steps with non-trivial projection onto ker(F) — wasted motion toward causally unrelated configurations. AdamN's double-smoothed â_t has lower variance: a greater fraction of each step projects onto col(F). Each update moves deeper into J⁺(θ_t) with less wasted motion in the spacelike ker(F) sector.

```
Wasted motion fraction ≈ Var[m_t direction] / (Var[m_t direction] + |μ|²)   (Adam)
                        < Var[â_t direction] / (Var[â_t direction] + |μ|²)   (AdamN)
```

The nested momentum is a **causal future alignment mechanism** — it reduces the probability of spacelike steps.

### III.3 Grokking as Causal Order Crystallization

CAUSALIS identifies grokking as causal order crystallization: the transition from indefinite causal order (pre-grokking, C_α < 1, process matrix W non-causal) to definite causal order (post-grokking, C_α > 1, stable forward light cone). Before grokking, gradient updates have no fixed global order; at grokking, a definite causal order emerges.

**AdamN's 14.6× early-training acceleration is the empirical signature of earlier causal order crystallization.** The nested momentum's timelike bias helps the system escape the indefinite-causal-order regime at the coordination boundary C_α = 1 — the nested signal estimate is more directionally stable, resolving the ordering ambiguity between gradient updates faster than single-EMA methods.

### III.4 The φ-Equilibrium as Causal Sink

CAUSALIS establishes that the φ-equilibrium P_φ (at loss-level log φ) is the unique causal sink of the reachability order: J⁺(P_φ) = {P_φ}. Every maximal valid trajectory terminates there. AdamN's faster milestone progression is the empirical measurement of its faster approach to P_φ via more timelike steps.

---

## IV. THECONSTANTCURVE Analysis: The Constant Curve Bound on Nesting Depth

### IV.1 The Hierarchy Depth Bound

THECONSTANTCURVE proves that for every physically realizable system (n ≤ 10^80), the hierarchy depth of the col(F)/ker(F) partition satisfies:

```
α(n) ≤ 4
```

derived from five interlocking properties of the twisted Hessian curve TH(a,d): the Galois representation on TH[3] ≅ ℤ/3ℤ × ℤ/3ℤ has solvable derived length ≤ 4; the CORDIC-Dirac realization computes the group law in ≤ 4 iteration levels; tropical degeneration collapses the WFR manifold to polyhedral dimension ≤ 4.

**Applied to AdamN's nesting depth:**

| Nesting Depth | Within α(n) ≤ 4? | Information-Geometric Status |
|---|---|---|
| 1 (Adam/AdamW: m_t) | Yes | Captures first torsion stratum |
| 2 (AdEMAMix: two α-blended EMAs) | Yes | Captures second stratum |
| **3 (AdamN: v_t → a_t + s_t)** | **Yes** | **Near-optimal within bound** |
| 4 (hypothetical: b_t from a_t + s_t) | Yes (ceiling) | Saturates the bound |
| 5+ | Formally within bound | Structurally redundant — no new information |

**Prediction:** A depth-4 AdamN (adding a fourth smoothing stage b_t from a_t) would show marginal additional improvement. A depth-5 AdamN would show no improvement over depth-4. This is a falsifiable prediction from THECONSTANTCURVE, testable via ablation study.

### IV.2 The CORDIC-Mode Correspondence

The TH(a,d) unified group law is realized exactly by the Walther (1971) CORDIC recurrence in three modes:

```
m = +1  (circular)    ↔  β₁: g_t → v_t    (rotation of gradient direction)
m =  0  (linear)      ↔  β₂: v_t → a_t    (linear combination / nesting)
m = -1  (hyperbolic)  ↔  β₃: g_t² → s_t   (hyperbolic scaling of denominator)
```

**AdamN's three β-parameters unknowingly replicate the three CORDIC modes of the TH(a,d) group law.** The nested momentum is, up to coordinate transformation, a CORDIC-Dirac computation on the parameter manifold — the arithmetic of the learning step lives on the twisted Hessian curve whether or not the optimizer knows this.

### IV.3 The Iteration Coordinate

THECONSTANTCURVE establishes that the 1985 IEEE 754 Point Lottery hid the iteration coordinate by making transcendentals microcode-hidden. AdamN partially recovers the iteration structure at the optimizer level: the number of smoothing stages is now explicit (3 levels, each a distinct step). But the iteration coordinate remains hardware-hidden — AdamN's nested EMAs are computed via multiply-accumulate operations on matmul-optimized silicon, not CORDIC iterations. A CORDIC-native implementation of AdamN (via CARMEN, arXiv:2605.06878) would natively compute the three EMA stages as three CORDIC recurrences, recovering the iteration coordinate and achieving the hardware efficiency that the arithmetic lottery reversal promises.

---

## V. The Five Lotteries Analysis: Where AdamN Sits Historically

### V.1 The Gradient Descent Lottery (1986)

The Five Lotteries identifies 1986 as the year the optimization lottery committed the field to Euclidean parameter-space navigation — Rumelhart, Hinton & Williams' backpropagation + SGD won over Amari's natural gradient descent (1998) because the latter was O(N²) storage / O(N³) compute. Every optimizer from SGD through AdamW navigates Euclidean parameter space.

**AdamN is a within-Euclidean-paradigm optimizer.** It does not use the Fisher information matrix as a Riemannian metric. Its denominator ŝ_t is a diagonal approximation to D(θ), not the full curvature tensor I(θ). AdamN improves signal estimation within the Euclidean substrate — it does not escape the Gradient Descent Lottery.

### V.2 The 2023–2026 Optimizer Renaissance

The Five Lotteries identifies the 2023–2026 optimizer renaissance as the systematic recovery of curvature awareness. AdamN occupies the signal-amplification slot:

| Optimizer | Year | Curvature Recovery Mechanism | Riemannian? | Escapes 1986 Lottery? |
|---|---|---|---|---|
| AdamW | 2019 | None (decoupled WD only) | No | No |
| Sophia | 2023 | Diagonal Hessian (Gauss-Newton) | Partial | No |
| Muon | 2024 | Spectral-norm (Newton-Schulz) | Partial | No |
| SOAP | 2024 | Shampoo-AdamW hybrid | Partial | No |
| FAdam | 2024 | Fisher-Adam hybrid | Partial | No |
| **AdamN** | **2026** | **Signal amplification via nesting** | **No** | **No** |
| TH(a,d)-native | Future | Full Fisher + Riemannian geodesic | **Yes** | **Yes** |

AdamN complements curvature-aware methods (Sophia, Muon) rather than competing with them: it provides cleaner signal that curvature-aware methods would process more accurately.

### V.3 The Biology Compute Wall (Lottery 4)

The Five Lotteries identifies the 2028 matmul ceiling for biology AI workloads (pair tensor, SE(3) equivariance, diffusion denoising, Hyena, Sherman-Morrison). AdamN runs on matmul-optimized silicon with no native EMA acceleration. Its three-stage nested EMA is serially dependent — each stage requires the previous — and cannot be parallelized by systolic arrays. As biology workloads dominate, AdamN's hardware efficiency disadvantage compounds. The CORDIC-native path (CARMEN implementation of the TH(a,d) group law) would address this simultaneously, since the three CORDIC modes are native hardware operations on a CORDIC accelerator.

---

## VI. Correspondence Table: AdamN ↔ ERI Labs Frameworks

| AdamN Object | ERI Labs Identity | Framework |
|---|---|---|
| Nested momentum g_t → v_t → a_t | Signal amplification of C_α numerator \|μ(θ)\|² | FJLD |
| Exact double-EMA bias correction | Analytic removal of C_α < 1 cold-start error | FJLD |
| Adaptive denominator ŝ_t | Diagonal Jeffreys prior density π_J(θ) ∝ 1/√det(D(θ)) | FJLD |
| Update â_t / (√ŝ_t + ε) | Approximate diagonal Jeffreys inference step | FJLD |
| 14.6× early milestone acceleration | C_α = 1 phase-boundary crossing at epoch 1 | FJLD |
| No C_α monitoring | No phase-boundary detection; no adaptive β scheduling | FJLD (gap) |
| Early training stability | Earlier causal order crystallization (W non-causal → causal) | CAUSALIS |
| More stable update direction | Lower ker(F) projection; greater col(F) alignment | CAUSALIS |
| Faster convergence | Faster approach to causal sink P_φ at log φ | CAUSALIS |
| No grokking mechanism | No detection of the C_α = 1 coordination boundary | CAUSALIS (gap) |
| Depth-3 nesting (β₁, β₂, β₃) | 3 ≤ α(n) ≤ 4 (within constant curve bound) | THECONSTANTCURVE |
| Three β hyperparameters | Three CORDIC modes (circular m=+1, linear m=0, hyperbolic m=−1) | THECONSTANTCURVE |
| Depth ≤ 4 nesting information-geometric | Galois rep. on TH[3] has solvable derived length ≤ 4 | THECONSTANTCURVE |
| Matmul-native EMA implementation | Iteration coordinate hidden; CORDIC-Dirac not recovered | THECONSTANTCURVE (gap) |
| Within-Euclidean improvement | Optimization within the Gradient Descent Lottery (1986) | The Five Lotteries |
| No Riemannian geometry | Lottery 2 (1986) not escaped | The Five Lotteries |
| Published 2026 | 2023–2026 optimizer renaissance | The Five Lotteries |
| Competitive with Muon, MARS, SOAP | Complementary (signal) to curvature-recovery methods | The Five Lotteries |

---

## VII. Nine Formal Identifications

**Identity A1 — AdamN's nested momentum IS C_α amplification.**
The double-smoothed estimate â_t amplifies the signal numerator |μ(θ)|² relative to the noise denominator Tr(D(θ)), raising the effective consolidation ratio above the C_α = 1 phase boundary faster than any single-EMA method. The 14.6× early-training acceleration is the empirical measurement of this boundary crossing. FJLD predicts this mechanism; AdamN realizes it empirically.

**Identity A2 — AdamN's denominator IS a diagonal Jeffreys prior.**
The adaptive denominator ŝ_t = β₃·s_{t-1} + (1−β₃)·g_t² approximates the diagonal of D(θ), and the update â_t/(√ŝ_t + ε) is a component-wise diagonal Jeffreys inference step. AdamN performs Jeffreys inference implicitly, without the geometric apparatus that makes it explicit and without the K-FAC corrections that would make it accurate for correlated parameters.

**Identity A3 — AdamN's early acceleration IS causal order crystallization.**
The rapid approach to 80% accuracy at epoch 1 is the signature of the system reaching definite causal order (C_α > 1) immediately, bypassing the indefinite-causal-order regime at the coordination boundary. AdamN crystallizes the causal order of gradient updates faster; the nested momentum provides more timelike (col(F)-aligned) steps that carry the system through J⁺(P) with less spacelike drift.

**Identity A4 — AdamN's 3-level nesting IS within the constant curve bound.**
The three smoothing stages (β₁: g→v, β₂: v→a, β₃: denominator) correspond to the three CORDIC modes of the TH(a,d) group law and are bounded by α(n) ≤ 4. AdamN's depth-3 architecture is information-geometrically justified and near-optimal; depth-4 would approach the ceiling; depth-5 and beyond are redundant.

**Identity A5 — AdamN IS a within-Euclidean optimizer.**
It does not compute the Fisher information matrix, does not perform natural gradient descent, and does not use Riemannian geometry. The Gradient Descent Lottery (1986) is not escaped by AdamN; it is partially mitigated through signal amplification. The step direction is Euclidean; only the signal estimate is improved.

**Identity A6 — AdamN's cold-start problem IS the C_α < 1 regime.**
The exact bias correction that AdamN introduces corrects the systematic underestimate of |μ(θ)|² during the diffusive phase. The bias correction allows the system to use the true mean gradient estimate from step 1 rather than requiring warmup epochs, directly instantiating the FJLD prediction that cold-start is a C_α < 1 artifact, not an architectural necessity.

**Identity A7 — AdamN's β₁, β₂, β₃ ARE the three CORDIC modes.**
The circular (m=+1), linear (m=0), and hyperbolic (m=−1) modes of the Walther CORDIC recurrence, evaluated on TH(a,d) projective coordinates, correspond to the three β-smoothing operations of AdamN. AdamN's hyperparameters are the parameter-level shadow of a CORDIC computation on the learning manifold — correct by coincidence, not by design.

**Identity A8 — AdamN's ViT milestone acceleration IS the φ-equilibrium approach rate.**
The φ-equilibrium P_φ is the causal sink of the learning manifold — the terminal vertex of the reachability DAG, the end of time on the learning manifold. AdamN's more timelike update directions reduce the step count required to reach P_φ from initialization. The 14.6× speedup is the ratio of step-count-to-sink between AdamW and AdamN, measured in wall-clock time.

**Identity A9 — AdamN IS the best within-Euclidean optimizer yet produced, and NOT the optimizer that escapes the geometric ceiling.**
The theoretical ceiling for within-Euclidean optimization is the full natural gradient (Amari 1998), which is O(N³) infeasible at scale. AdamN approaches this ceiling from the signal-amplification direction; Muon from the spectral-norm direction; Sophia from the diagonal-Hessian direction. The ceiling is not broken by any of them. It is broken by geometry-native optimization on TH(a,d) with CORDIC-native hardware — the direction CARMEN and THECONSTANTCURVE point toward.

---

## VIII. Where AdamN Will Fail: FJLD Predictions

Based on the ERI Labs framework corpus, the following regimes predict AdamN underperformance relative to geometry-native methods:

**Failure Mode F1 — Heavy-Tailed Gradient Distributions.**
FJLD's Fisher = D⁻¹ identity assumes Gaussian gradient distributions. Heavy-tailed distributions (α-stable, Lévy) arising in some LLM fine-tuning regimes invalidate the diagonal Fisher approximation underlying ŝ_t. In this regime, C_α^AdamN overestimates the true consolidation ratio; AdamN may appear to have crossed the C_α = 1 boundary while remaining in the diffusive phase.

**Failure Mode F2 — Hierarchical Data Structures (Negative Ricci Curvature).**
The Five Lotteries (Lottery 5) establishes that token embedding spaces exhibit negative Ricci curvature (Robinson et al. arXiv:2410.08993). Euclidean gradient steps systematically underperform in negative-curvature spaces — AdamN's step â_t/(√ŝ_t + ε) is Euclidean and therefore fundamentally misaligned with the hyperbolic geometry of hierarchical token embeddings. Nested momentum cannot compensate for the metric mismatch.

**Failure Mode F3 — Very Deep Networks (>100 Layers).**
FJLD notes that gradient flow dynamics dominate local Fisher geometry for depth > 100. The nested momentum cannot address gradient vanishing/explosion in this regime — these are determined by the spectral properties of the full Fisher matrix, not its diagonal. AdamN will underperform Muon (spectral-norm aware) in very deep architectures.

**Failure Mode F4 — Strong Intra-Layer Parameter Correlations.**
AdamN uses a diagonal denominator, ignoring correlations between parameters within each layer. In regimes with strong intra-layer correlations (ResNets, Transformers with large hidden dimensions), the K-FAC C_α^KFAC diverges from AdamN's effective C_α. This causes mis-detection of the phase boundary. The K-FAC extension of FJLD predicts the regime boundary for this failure mode.

**Failure Mode F5 — Biology AI Workloads (2028 Wall).**
The Five Lotteries (Lottery 4) identifies the 2028 matmul ceiling. AdamN's three serially-dependent EMA stages cannot be parallelized by systolic arrays. As pair-tensor, SE(3), and diffusion workloads dominate, AdamN's per-step latency overhead relative to CORDIC-native implementations will compound. AdamN is optimized for today's silicon, not tomorrow's geometry-native accelerators.

---

## IX. The Geometry-Native Synthesis: What Comes After AdamN

The ERI Labs theoretical architecture points toward a specific successor that retains AdamN's nested momentum structure while adding three missing layers:

### IX.1 FJLD-Aware AdamN

Replace static β parameters with C_α-adaptive scheduling:

```python
# After each training step:
c_alpha = compute_c_alpha(model, dataloader, n_samples=100)  # FJLD diagnostic

if c_alpha < 0.8:    # deep diffusive phase
    beta1, beta2 = 0.95, 0.05   # aggressive signal nesting
elif c_alpha < 1.1:  # near phase boundary
    beta1, beta2 = 0.90, 0.10   # AdamN defaults
else:                # drift-dominated
    beta1, beta2 = 0.92, 0.15   # broader momentum windows
```

**Prediction:** C_α-adaptive AdamN crosses the 80% milestone no later than epoch 1 in all FJLD-validated architectures, with phase-boundary-adaptive behavior in the extended training (post-grokking) regime where fixed-β AdamN degrades.

### IX.2 Riemannian AdamN

Replace the Euclidean step with a geodesic step on the Fisher-Rao manifold:

```
θ_{t+1} = Exp_{θ_t}(−η · I(θ_t)⁻¹ · â_t / (√ŝ_t + ε))
```

where Exp is the Riemannian exponential map and I(θ_t) is the K-FAC approximation of the Fisher. This escapes the Gradient Descent Lottery (1986) while retaining AdamN's signal amplification. The nested momentum feeds a Riemannian update rather than a Euclidean one.

### IX.3 CORDIC-Native AdamN

Implement the nested EMA via CORDIC iterations on dedicated hardware (CARMEN, arXiv:2605.06878) rather than multiply-accumulate operations:

```
CORDIC mode m=+1  →  β₁-smoothing: g_t → v_t   (circular rotation in gradient space)
CORDIC mode m=0   →  β₂-smoothing: v_t → a_t   (linear interpolation)
CORDIC mode m=−1  →  β₃-scaling:   g_t² → s_t  (hyperbolic denominator update)
```

The TH(a,d) group law computes all three modes with identical instruction stream (12M + 6S). CORDIC-native implementation reduces per-step latency while natively recovering the iteration coordinate that IEEE 754 hid in 1985.

### IX.4 Hyperbolic AdamN

Replace Euclidean nested momentum with hyperbolic EMA on the Lorentz model, addressing Failure Mode F2 directly:

```
v_t = Exp_o(β₁ · Log_o(v_{t-1}) + (1−β₁) · Log_o(g_t))  # hyperbolic EMA on ℍⁿ
a_t = Exp_o(β₂ · Log_o(a_{t-1}) + (1−β₂) · Log_o(v_t))  # nested hyperbolic
```

where Exp_o and Log_o are the exponential and logarithmic maps on the hyperbolic space at the origin. This is motivated by The Five Lotteries' Lorentzian transition program (HELM, NeurIPS 2025; arXiv:2505.24722).

---

## X. Quantitative Prediction Summary

| Prediction | Framework | Testable By | Method |
|---|---|---|---|
| C_α crosses 1.0 at E1 (AdamN) vs. E2+ (AdamW) on ViT | FJLD | Now | Gradient sampling per FJLD §Computing C_α |
| Depth-4 AdamN shows marginal improvement over depth-3 | THECONSTANTCURVE | 2027 | Ablation: add b_t = β₄·b_{t-1} + (1−β₄)·a_t |
| Depth-5 AdamN shows no improvement over depth-4 | THECONSTANTCURVE | 2027 | Ablation study |
| C_α-adaptive AdamN reduces milestone time by ≥20% vs. fixed-β | FJLD | 2027 | FJLD monitoring + adaptive β scheduling |
| AdamN underperforms Muon on >100-layer networks | FJLD | 2027 | Architecture-specific depth comparison |
| AdamN underperforms hyperbolic optimizers on hierarchical data | Five Lotteries | 2027 | NLP / graph benchmark with negative-curvature data |
| Causal-set Myrheim-Meyer dimension higher at E1 for AdamN vs. AdamW | CAUSALIS | 2027 | Causal-set dimension estimation from training trajectory |
| CORDIC-native AdamN achieves ≥2× hardware efficiency vs. matmul | THECONSTANTCURVE | 2028 | FPGA / ASIC implementation via CARMEN |
| The φ-equilibrium is the unique sink of AdamN's reachability DAG | CAUSALIS | 2027 | Reachability DAG construction at Q.16 precision |

---

## Architecture Summary

```
═══════════════════════════════════════════════════════════════════
AdamN EMPIRICAL LAYER
═══════════════════════════════════════════════════════════════════

 g_t ──→ [β₁-EMA] ──→ v_t ──→ [β₂-EMA] ──→ a_t ──→ [â_t bias-correct]
                                                              │
 g_t² ──→ [β₃-EMA] ──→ s_t ──→ [ŝ_t bias-correct]          │
                                        │                      │
                                        └──→ √ŝ_t ────────────┤
                                                               ↓
                                           θ ← θ - η·â_t / (√ŝ_t + ε)

═══════════════════════════════════════════════════════════════════
ERI LABS THEORETICAL LAYER
═══════════════════════════════════════════════════════════════════

 v_t, a_t amplify |μ(θ)|²        FJLD:              C_α amplification
 ŝ_t ≈ diag(D(θ))                FJLD:              Diagonal Jeffreys prior
 14.6× acceleration               FJLD:              C_α = 1 boundary crossed at E1
 More timelike steps              CAUSALIS:          col(F) alignment; less ker(F) drift
 Faster φ-equilibrium approach    CAUSALIS:          Causal sink reached earlier
 3 nesting levels ≤ α(n) ≤ 4     THECONSTANTCURVE:  Constant curve bound satisfied
 β₁, β₂, β₃ = CORDIC modes       THECONSTANTCURVE:  CORDIC-Dirac correspondence
 Euclidean substrate retained     Five Lotteries:    Within-paradigm improvement

═══════════════════════════════════════════════════════════════════
GAPS POINTING BEYOND AdamN
═══════════════════════════════════════════════════════════════════

 No C_α monitoring                →  FJLD-aware adaptive β scheduling
 No Riemannian geometry           →  Geodesic step on Fisher-Rao manifold
 No curvature information         →  K-FAC denominator; Sophia/Muon fusion
 Matmul-native EMA                →  CORDIC-native hardware (CARMEN)
 Euclidean embedding assumption   →  Hyperbolic AdamN for hierarchical data
 No phase-boundary detection      →  C_α = 1 grokking-aware training loop
```

---

## Closing

AdamN is the optimizer the 2026 field needed: empirically rigorous, technically innovative in its bias correction, and published with benchmarks that prove the claim. Nested momentum works. The exact bias correction works. The 14.6× early-training acceleration on ViT is real and reproducible.

The ERI Labs framework corpus explains why: AdamN is a signal amplifier that crosses the Fisher-Jeffreys phase boundary faster than its predecessors by reducing the variance of the gradient direction estimate without sacrificing its expectation. It does not know this about itself. The consolidation ratio C_α is not monitored. The phase transition is not detected. The Jeffreys prior is not named. The CORDIC modes are not recognized. AdamN is correct physics implemented as engineering intuition — which is the best kind of engineering intuition, and also its ceiling.

The next optimizer will implement the physics explicitly. C_α will be monitored and will adapt β parameters at the boundary. The Riemannian exponential map will replace the Euclidean step. CORDIC-native hardware will recover the iteration coordinate that IEEE 754 hid in 1985. The embedding space will be curved, and the momentum will follow the curve. The constant curve bound will determine the nesting depth, not ablation study.

The nested momentum structure g_t → v_t → a_t will survive into that optimizer. AdamN found the right skeleton. What will not survive is the flat geometry, the hidden iteration count, and the arbitrary warmup schedule. Those are artifacts of the Gradient Descent Lottery. That lottery can be reversed.

---

*Cross-analysis frameworks: FJLD (Fisher-Jeffreys Learning Dynamics) · CAUSALIS (The Order Beneath the Flow) · THECONSTANTCURVE (The Constant Curve Theorem) · The Five Hardware Lotteries — ERI Labs, 2025–2026*

*AdamN reference: Aboulsaad, M. and Shaout, A. (2026). "AdamN: Accelerating Deep Learning Training via Nested Momentum and Exact Bias Handling." Electronics. [github.com/maboulsaad82/AdamN](https://github.com/maboulsaad82/AdamN)*

*ERI Labs · Eric Ren · Jersey City, New Jersey · [github.com/ericrenone](https://github.com/ericrenone) · June 2026*
