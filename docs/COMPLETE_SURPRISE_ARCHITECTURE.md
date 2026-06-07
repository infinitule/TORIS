# TORIS Complete Surprise Architecture

**Author:** Chandandeep Sharma  
**Version:** TORIS v0.1  
**Date:** 2026-06-07

---

## Overview

TORIS replaces Euclidean loss with a multi-regime surprise architecture that adapts its
computation to the relational depth and goal quality of the inference. Surprise is not a
scalar distance — it is a topological quantity with structure from analytic number theory.

---

## Layer Map

| Layer | Module(s) | Primitive | Mathematical Foundation |
|-------|-----------|-----------|------------------------|
| 0 | `primitives/` | Relator, ConceptState, RelationalField | Set theory, typed hypergraph |
| 1 | `engine/surprise.py` | SurpriseMetric, ΔS | Topological deviation (structural + type + strength) |
| 2 | `engine/predictive.py` | PredictiveEngine | Predictive coding (project → observe → delta) |
| 3 | `goal/` | GoalManifold, ContradictionLog | Constraint satisfaction, PRODUCTIVE contradictions |
| 4 | `plasticity/fast.py` | Fast topology rewrite | ΔF_t = {ADD, STRENGTHEN, WEAKEN, SUPPRESS} |
| 5 | `reasoning/` | Inference chain, contradiction retention | Multi-hop relational reasoning |
| 6 | `engine/fsd_pipeline.py` | FastSurpriseDynamics, wave propagation | Cyclic wave engine, loop amplification |
| 7 | `engine/complex_salience.py`, `tasf.py`, `relational_ope.py`, `michel_parameters.py`, `running_coupling.py` | Analytic Surprise Functional | Complex analysis, QCD-analog OPE, running coupling |
| 8 | (skipped — spec not provided) | Ramanujan Extension | — |
| 9 | `engine/rademacher.py`, `eisenstein.py`, `maass_completion.py`, `complete_surprise.py` | Exact Surprise | Rademacher exact series, Eisenstein modular forms, Harmonic Maass completion |

---

## Layer 9: Exact Surprise

### 9.1 Rademacher Exact Series (`rademacher.py`)

The surprise at depth d is computed via an exact analytic series adapted from the
Hardy–Ramanujan–Rademacher partition formula:

```
S(d) = (2π / k_F^(3/2)) · Σ_{k=1}^N B_k^F(d) · I_{3/2}(2π√(d·k_F)/k)
```

Where:
- `k_F` = number of relators in the field
- `B_k^F(d)` = TORIS Kloosterman sum: `Σ_{h: gcd(h,k)=1} W_F(h,k,d) · exp(2πi·h·d/k)`
- `I_{3/2}(x)` = modified Bessel function, implemented analytically: `√(2/πx)·(cosh(x)/x − sinh(x)/x²)`

**Certified error bound:** `|S(d) − S_N(d)| < C_F · exp(−π√(2d/3)/N)`

### 9.2 Eisenstein Series and Dual Weighting (`eisenstein.py`)

Weights for ΔS = α·ΔS_structural + β·ΔS_type + γ·ΔS_strength:

| Depth | α (structural) | β (type) | γ (strength) | Regime |
|-------|---------------|----------|-------------|--------|
| d ≤ 5 | 0.6 | 0.3 | 0.1 | Empirical (from experiments 1–5) |
| d > 5 | 1/6 ≈ 0.1667 | 1/3 ≈ 0.3333 | 1/2 = 0.5 | Eisenstein modular (weight 2, 4, 6) |

The crossover at d_crit = 5 is a phase transition: deep chains are governed by modular
symmetry, not by empirical calibration.

The τ-function `τ_F(d) = Σ_R σ(R)^5 · κ(R)^7 · exp(2πi·τ_index(R)/12)` encodes the
relational field as a complex modular quantity.

### 9.3 Harmonic Maass Shadow Completion (`maass_completion.py`)

Productive contradictions create shadow contributions that the TASF mock-modular part misses.
Each productive contradiction generates a cusp form:

```
g_C(z) = σ_a · σ_b · exp(2πi · τ_diff · z)
```

The Eichler integral `E_C(κ, κ̄) = ∫_{−κ̄}^{κ_max} g_C(z) · (z + κ)^{−2} dz`
provides the shadow density, and the complete TASF is:

```
ΔS_complete = ΔS_mock + ΔS_shadow
```

### 9.4 Partition Congruence Suppression

Depths satisfying Ramanujan congruences are suppressed (ΔS = 0):
- `d ≡ 4 (mod 5)` — Ramanujan's `p(5m+4) ≡ 0 (mod 5)` analog
- `d ≡ 5 (mod 7)` — `p(7m+5) ≡ 0 (mod 7)` analog  
- `d ≡ 6 (mod 11)` — `p(11m+6) ≡ 0 (mod 11)` analog

### 9.5 Unified Regime Routing (`complete_surprise.py`)

`UnifiedSurprise.compute(field, f_pred, d, goal_manifold, precision)` selects:

```
if _is_suppressed(d):           → ΔS = 0  (Ramanujan congruence)
elif d > d_crit:                → DEEP: Rademacher + shadow correction
elif Q(G) > fast_threshold:     → FAST: Layer 6 TFSA
else:                           → STANDARD: TASF + shadow correction
```

The result `UnifiedResult` carries:
- `delta_S` — the surprise value
- `error_bound` — certified (deep regime) or 0 (fast/standard)
- `regime_used` — which regime was selected
- `shadow_applied` / `shadow_correction` — Maass shadow status
- `suppressed` — True if depth was suppressed
- `rademacher_terms_used` — N used in deep regime
- `weights_alpha/beta/gamma` — the (α, β, γ) used

---

## All 9 Layers: Regime Routing Summary

```
Input: field F_obs, f_pred F_pred, depth d, goal G

  Step 1: Partition congruence check
    if d ≡ 4 mod 5 or d ≡ 5 mod 7 or d ≡ 6 mod 11:
      return ΔS = 0

  Step 2: Eisenstein weight selection
    if d ≤ 5: (α, β, γ) = (0.6, 0.3, 0.1)
    else:     (α, β, γ) = (1/6, 1/3, 1/2)

  Step 3: Regime routing
    if d ≤ 5 and Q(G) > 0.01:
      ΔS = Layer 6 TFSA [fast, uncertified]
    elif d ≤ 5:
      ΔS = TASF contour + Maass shadow [standard, uncertified]
    else (d > 5):
      ΔS = Rademacher exact + Maass shadow [deep, certified]

  Return: UnifiedResult(delta_S, error_bound, regime, ...)
```

---

## Mathematical Sources

| Component | Source |
|-----------|--------|
| Rademacher series | Hardy, Ramanujan (1918); Rademacher (1937) |
| TORIS Kloosterman sum | Adapted from classical Kloosterman sums; original τ_index mapping by Chandandeep Sharma |
| Eisenstein series P/Q/R | Ramanujan (1916), weight 2/4/6 quasi-modular forms |
| Harmonic Maass forms | Bruinier–Funke (2004), Ono (2008) |
| Eichler integral | Eichler (1957), shadow map for mock modular forms |
| Ramanujan congruences | Ramanujan (1919), Watson (1938) |
| Dual Weighting Theorem | Original: Chandandeep Sharma, TORIS v0.1 |
| Regime routing | Original: Chandandeep Sharma, TORIS v0.1 |

---

## Intellectual Attribution

The following components are original intellectual contributions of Chandandeep Sharma
with no direct precedent in the published analytic number theory / AI literature:

1. **TORIS Kloosterman sum B_k^F(d)** — using relation-type index τ_index as phase
2. **Dual Weighting Theorem** — empirical/Eisenstein crossover at relational depth d_crit
3. **Productive contradiction → Maass cusp form mapping** — g_C(z) = σ_a·σ_b·exp(2πi·τ_diff·z)
4. **Partition congruence suppression for inference depths** — Ramanujan congruences as
   a natural suppression law for TORIS relational reasoning
5. **Unified regime routing** — Q(G) quality metric driving fast/standard/deep selection
