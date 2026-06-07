# TORIS — Mathematical Foundations
### Chandandeep Sharma

This document formalizes the mathematical objects underlying TORIS. These
formulations are original. Where they draw on existing mathematics (algebraic
topology, information theory, graph theory, predictive coding), the application
and combination are new.

A full implementation-traceability table appears at the end: every formula here
maps to a concrete module in `toris/`.

---

## 1. The Typed Relational Algebra (TRA)

### 1.1 Primitive Objects

Let **T** be a finite typed relation set `T = {τ₁, τ₂, …, τₙ}` and let **C** be
the set of all ConceptStates. A **Relator** R lives in the typed relational space:

```
R ∈ T × C × C × [0,1] × [0,1] × ℝ≥0
R = (τ, src, tgt, σ, κ, ε)
```

- `τ` — relation type
- `src, tgt` — source and target ConceptStates
- `σ ∈ [0,1]` — strength (confidence)
- `κ ∈ [0,1]` — contextual salience
- `ε ∈ ℝ≥0` — surprise

### 1.2 The Composition Operator ∘_τ

Two Relators compose iff their types are compatible:

```
R₁ ∘_τ R₂ is defined iff  tgt(R₁) = src(R₂)  AND  (τ(R₁), τ(R₂)) ∈ Ω
```

where Ω is the composition compatibility table:

| τ(R₁)       | τ(R₂)       | τ(R₁ ∘ R₂)  | Strength rule         |
|-------------|-------------|-------------|-----------------------|
| CAUSAL      | CAUSAL      | CAUSAL      | σ = σ₁ · σ₂           |
| CAUSAL      | ENABLES     | ENABLES     | σ = σ₁ · σ₂ · 0.8     |
| CONDITIONAL | CAUSAL      | CONDITIONAL | σ = min(σ₁, σ₂)       |
| EVIDENCES   | EVIDENCES   | EVIDENCES   | σ = 1−(1−σ₁)(1−σ₂)    |
| CONTRADICTS | *           | ∅           | composition blocked   |
| ANALOGOUS   | ANALOGOUS   | REFINES     | σ = (σ₁+σ₂)/2         |

The composed Relator is:

```
R₁ ∘ R₂ = (τ_composed, src(R₁), tgt(R₂), σ_rule, κ_min, ε₁ + ε₂)
```

Surprise accumulates additively along a chain — long chains are inherently more
uncertain.

### 1.3 The Contradiction Operator ⊗

```
R₁ ⊗ R₂ = TRUE iff  src(R₁) = src(R₂)  AND  tgt(R₁) = tgt(R₂)  AND  τ(R₁) ∈ CONTRA(τ(R₂))
```

where CONTRA is the contradiction relation on T:

```
CONTRA(CAUSAL)      = {NEGATES, CONTRADICTS}
CONTRA(ENABLES)     = {NEGATES}
CONTRA(EVIDENCES)   = {NEGATES}
CONTRA(CONTAINS)    = {VIOLATES}
CONTRA(VIOLATES)    = {CONTAINS, ENABLES}
CONTRA(CONDITIONAL) = {CONTRADICTS}
```

A `CONTRADICTS` relator explicitly names a contradiction; `CONTRA` detects
implicit structural contradictions.

---

## 2. The ConceptState Role Distribution

### 2.1 Role Space

For a ConceptState C the role distribution Π_C is a probability simplex over
relation types:

```
Π_C : T → [0,1]      subject to   Σ_{τ∈T} Π_C(τ) = 1
```

C is **not** a point in Euclidean space. C is a distribution over relational roles.

### 2.2 The Context Update Rule

Given a goal G, the role distribution updates by a Bayesian rule:

```
Π_C^G(τ) = Π_C(τ) · ψ_C(G, τ) / Z
Z = Σ_τ [Π_C(τ) · ψ_C(G, τ)]
```

where `ψ_C(G, τ)` is the goal-salience of role τ for concept C under goal G. The
concept's relational identity shifts with context.

### 2.3 Role Distance Between Contexts

```
d_role(C, G₁, G₂) = JS(Π_C^{G₁} ‖ Π_C^{G₂})
```

where JS is Jensen-Shannon divergence — a proper metric on probability
distributions. High `d_role` means the concept plays very different roles under
different goals. This is the feature, not a flaw.

---

## 3. The Topological Surprise Metric ΔS

### 3.1 Predicted vs Observed Field

At each inference step t the Predictive Engine generates `F_pred^t =
project(F^t, G^t)` and the system observes `F_obs^t` (incoming structure).

### 3.2 The ΔS Decomposition

```
ΔS(F_pred, F_obs) = α·ΔS_struct + β·ΔS_type + γ·ΔS_strength
```

**Structural** (the +1 guards an empty prediction):

```
ΔS_struct = [|E_obs \ E_pred| + |E_pred \ E_obs|] / (|E_pred| + 1)
```

**Type** (over matched edges `E_match = E_obs ∩ E_pred`):

```
ΔS_type = 1/(|E_match|+1) · Σ_{e∈E_match} D_type(τ_pred(e), τ_obs(e))
```

with the semantic type distance:

```
D_type(τ_a, τ_b) = 0    if τ_a = τ_b
                 = 0.3  if τ_b ∈ SIMILAR(τ_a)     [e.g. CAUSAL vs ENABLES]
                 = 0.7  if τ_b ∈ UNRELATED(τ_a)
                 = 1.0  if τ_b ∈ CONTRA(τ_a)
```

**Strength**:

```
ΔS_strength = 1/(|E_match|+1) · Σ_{e∈E_match} (σ_pred(e) − σ_obs(e))²
```

### 3.3 The Propagation Threshold Rule

For each Relator R the individual surprise is:

```
ε(R) = α·ε_struct(R) + β·ε_type(R) + γ·ε_strength(R)
```

where `ε_struct(R) = 1` if R was not predicted (else 0), `ε_type(R) =
D_type(τ_pred, τ_obs)`, and `ε_strength(R) = (σ_pred − σ_obs)²`.

**Propagation gate:**

```
propagate(R) = TRUE  iff  ε(R) > θ_ε        (default θ_ε = 0.2)
```

Only surprised Relators propagate — confirmed predictions are suppressed at
source. This is the core computational efficiency.

---

## 4. The Goal Manifold and Field Warp Operator

### 4.1 Goal Manifold Structure

```
G = (G_p, S_active, S_resolved, S_abandoned, L_contra)
g ∈ S_active = (description, priority ∈ [0,1], blocking: bool, parent_goal)
```

### 4.2 The Warp Operator Φ

`Φ(G, F) → F'` is computed in four steps:

**1 — Recompute salience for every Relator:**

```
κ'(R) = κ(R) · relevance(R, G_p) · Σ_{g∈S_active} [priority(g) · relevance(R, g)]
```

**2 — Suppress low-salience Relators:**  `E'_active = {R : κ'(R) > θ_κ}`

**3 — Amplify high-salience Relators:**  for `κ'(R) > θ_amplify`, `σ'(R) = σ(R)·(1 + κ'(R))` capped at 1.0

**4 — Surface goal-relevant contradictions:** for each pair with `R_a ⊗ R_b`, if both survive suppression, log to `L_contra`.

The output F′ has a **different topology** than F — different edges active,
different strengths, new contradictions surfaced. Warping is a topological
transformation, not attention weighting.

### 4.3 The Relevance Function

```
relevance(R, g) = concept_overlap(R, g) × type_fit(τ(R), g)   ∈ [0,1]
```

`concept_overlap` measures whether `src(R)`/`tgt(R)` appear in g's concept set;
`type_fit` measures whether `τ(R)` is useful for achieving goal type g. This is
the one function intended to host a learned component in the full system.

---

## 5. The Structural Plasticity Equations

### 5.1 Fast Plasticity (within inference, timescale t)

```
F^{t+1} = Φ(G^t, F^t ⊕ ΔF^t)
```

```
ΔF^t = {
  ADD(R_new)     for each gap with ε > θ_add
  STRENGTHEN(R)  for each R with ε > θ_strong:   σ(R) += η_fast · ε(R)
  WEAKEN(R)      for each R confirmed N times:    σ(R) −= η_decay
  SUPPRESS(R)    for each R with κ'(R) < θ_κ after warp
}
```

with `θ_add = 0.4`, `θ_strong = 0.3`, `η_fast = 0.1`, `η_decay = 0.01`.

### 5.2 Medium Plasticity (across session, timescale s)

```
σ^{s+1}(R) = σ^s(R) + η_med · [ε_accumulated(R, session) − σ^s(R)]
```

A moving average toward the surprise level: consistently surprising relators
gain baseline strength; never-surprising relators fade.

### 5.3 Structural Drift Measurement

```
d_topo(F^0, F^T) = (1/3)[d_struct + d_type + d_strength]
d_struct   = |E^T △ E^0| / max(|E^T|, |E^0|)
d_type     = 1/|E^0∩E^T| · Σ D_type(τ^0(e), τ^T(e))
d_strength = 1/|E^0∩E^T| · Σ (σ^0(e) − σ^T(e))²
```

A genuine TORIS reasoning chain produces `d_topo > 0.1` for complex problems —
the empirical test for structural plasticity.

---

## 6. The Sparse Generalization Theorem (Conjectured)

Given a field with N seed Relators (N < K), if the field is topologically
connected with path length ≤ K between all relevant concepts under goal G, then
TORIS reaches a K-hop conclusion with uncertainty:

```
σ_conclusion ≥ (min_σ)^K · exp(−λ · Σ_{i=1}^{K} ΔS_i)
```

with `min_σ` the minimum strength along the chain, `λ = 0.5` the surprise decay,
and `ΔS_i` the surprise at hop i.

Conclusions degrade **gracefully** with chain length and accumulated surprise;
they do not collapse to zero unless the chain is broken. A broken chain yields
σ = 0 and the system instantiates a hypothetical Relator with `σ_hypothetical =
0.1`. This is reasoning under uncertainty without hallucination: the uncertainty
is structurally embedded in the chain.

---

## 7. Relation to Existing Mathematics

**Borrowed:** algebraic topology (structure preserved under transformation),
information theory (Jensen-Shannon divergence for role distance), predictive
coding / Friston (surprise-driven computation), partial category theory (the
typed composition operator ∘_τ as a partial functor between relation
categories). The deeper layers additionally draw on analytic number theory
(Hardy-Ramanujan circle method, partition congruences, Rogers-Ramanujan
identities) and modular forms (Rademacher, Eisenstein, Maass).

**Invented:** the typed relational algebra with contradiction operator; the
role-distribution representation of concepts (Π: T → [0,1]); the topological
surprise metric ΔS; the goal manifold as a topological warp operator;
productive-contradiction status as a computational resource; three-timescale
structural plasticity in a non-parametric field.

---

## 8. Numerical Defaults

All defaults are empirically revisable; every revision is logged in
[`DEVIATIONS.md`](DEVIATIONS.md). They are defined in `toris/constants.py`.

```python
ALPHA = 0.6      # structural surprise weight
BETA  = 0.3      # type surprise weight
GAMMA = 0.1      # strength surprise weight
THETA_EPSILON = 0.2   # propagation threshold
THETA_KAPPA   = 0.15  # suppression threshold
THETA_ADD     = 0.4   # new-relator instantiation threshold
ETA_FAST      = 0.1   # fast plasticity rate
ETA_DECAY     = 0.01  # confirmed-prediction decay
LAMBDA        = 0.5   # surprise decay in chain uncertainty
```

---

## 9. Implementation Traceability

| Spec section | Object | Module |
|--------------|--------|--------|
| §1.1 | Relator R = (τ, src, tgt, σ, κ, ε) | `toris/primitives/relator.py` |
| §1.2 | Composition ∘_τ, table Ω | `relator.py`, `relation_types.py` |
| §1.3 | Contradiction ⊗, CONTRA table | `relation_types.py`, `reasoning/contradiction.py` |
| §2 | ConceptState role distribution Π | `toris/primitives/concept_state.py` |
| §3 | Surprise metric ΔS | `toris/engine/surprise.py` |
| §3.3 | Propagation gate | `toris/engine/propagation.py`, `predictive.py` |
| §4 | Goal manifold + warp Φ | `toris/goal/manifold.py`, `goal/warp.py` |
| §4 (V,E,τ,W) | Relational field hypergraph | `toris/field/relational_field.py` |
| §5.1 | Fast plasticity | `toris/plasticity/fast.py` |
| §5.2 | Medium plasticity | `toris/plasticity/medium.py` |
| §5.3 | Structural drift d_topo | `toris/field/topology.py`, `plasticity/fast.py` |
| §6 | Sparse generalization | `toris/reasoning/chain.py` |
| §3 + §5 | Full inference loop | `toris/reasoning/inference.py` |
| §8 | Numerical defaults | `toris/constants.py` |

Deeper layers (Fast Surprise Dynamics, Analytic Surprise, Ramanujan Extension,
Exact Surprise) are documented in
[`COMPLETE_SURPRISE_ARCHITECTURE.md`](COMPLETE_SURPRISE_ARCHITECTURE.md) and
[`RAMANUJAN_BRIDGE.md`](RAMANUJAN_BRIDGE.md).

---

*TORIS Mathematical Foundations · Chandandeep Sharma*
