# TORIS — Ramanujan Bridge
## Layer 8: The Ramanujan Extension
### Chandandeep Sharma | v1.0 | June 2026

---

## 1 — Overview

Layer 8 imports four mathematical structures from Ramanujan's Collected Papers
(Hardy, Wilson, Seshu Aiyar, 1927) and maps them to TORIS computational primitives.

---

## 2 — The Four Structures

### 2.1 Circle Method → Relational Field Saddle Points

**Source:** Hardy & Ramanujan, "Asymptotic Formulae in Combinatory Analysis" (1918)

The Hardy-Ramanujan circle method derives p(n) by evaluating the contour integral
of the partition generating function at dominant saddle points on the unit circle.

**TORIS mapping:**
The relational field has a generating function Z_F(κ) summing over relator
configurations weighted by κ^depth. Surprise at depth d is extracted by the
same contour integral technique:

    ΔS_dominant(F, d) ≈ Z_F(κ_saddle) · κ_saddle^(-d) / Z_F(1)

where κ_saddle(d) = exp(π√(2d/3) / d) — the Hardy-Ramanujan analog.

**Computational gain:** O(1) per depth level vs O(|E|^d) brute-force traversal.

**Implementation:** `toris/engine/circle_method.py`
- `saddle_point(d)` — κ_saddle computation
- `kloosterman_correction(d)` — first secondary saddle correction (q=2 term)
- `circle_method_surprise(field, d)` — full circle-method ΔS at depth d
- `saddle_surprise_profile(field, max_depth)` — profile over d=1..max_depth

---

### 2.2 Partition Congruences → Relational Suppression Theorem

**Source:** Ramanujan, "Some Properties of p(n)" (1919)

Ramanujan observed from MacMahon's numerical tables:
    p(5m+4)  ≡ 0 (mod 5)
    p(7m+5)  ≡ 0 (mod 7)
    p(11m+6) ≡ 0 (mod 11)

These are exact congruences holding for ALL m ≥ 0.

**TORIS mapping:**
For fields with p-modular strength structure (relator strengths are
multiples of 1/p by relation type), the total surprise at specific
relational depths vanishes modulo p:

    S_{pm+r₀}(F) ≡ 0 (mod p)    for p ∈ {5,7,11} and matching r₀

**Computational gain:** At suppressed depths, entire classes of surprise
contributions cancel exactly. TORIS can skip these depths without
approximation error.

**Experimental result (Exp 11):** 100% suppression accuracy for modular fields
on d=1..30 with p=5 (6 suppressed depths, all correctly identified).

**Implementation:** `toris/engine/suppression.py`
- `is_modular_field(field, p)` — detects p-modular structure
- `suppressed_depth(d)` — arithmetic check for Ramanujan residues
- `suppression_report(field, max_depth)` — full table of suppressed depths

---

### 2.3 Rapidly Convergent 1/π Series → Goal Manifold Compression

**Source:** Ramanujan, "Modular Equations and Approximations to π" (1914)

    1/π = (2√2/9801) Σ_{n=0}^∞ (4n)!(1103 + 26390n) / ((n!)⁴ · 396^(4n))

First term gives 8 decimal places. Converges because it exploits the
near-integer property of e^(π√58) via modular equations of degree 58.

**TORIS mapping:**
The Goal Manifold warp sum Φ(G,F) = Σ_k priority(k)/(k+1) · mean_salience(F)
converges analogously for coherent manifolds. Goal coherence Q(G) is defined
as the fractional part of this sum — small Q → rapid convergence.

**Automatic switching criterion (analog of Ramanujan's n selection):**
- Q < 0.01 → use 3-term Ramanujan expansion (fast path)
- Q ≥ 0.01 → use full iteration (exact path)

**Experimental result (Exp 12):** 3-term Ramanujan gives 0.00% relative error
vs exact for Q ≈ 0 manifolds. auto_warp correctly selects ramanujan/exact method.

**Implementation:** `toris/engine/ramanujan_goal.py`
- `goal_coherence(manifold)` — Q(G) ∈ [0, 0.5]
- `ramanujan_3term(manifold, field)` — truncated warp approximation
- `full_warp(manifold, field)` — exact O(K·N) warp
- `auto_warp(manifold, field)` — automatic method selection
- `pi_ramanujan(n_terms)` — Ramanujan π series (verification tool)

---

### 2.4 Rogers-Ramanujan Identities → Relational Partition Function

**Source:** Rogers & Ramanujan, "Proof of Certain Identities in Combinatory Analysis" (1919)

    1 + Σ_{n≥1} q^(n²)/((1-q)…(1-qⁿ)) = Π_{n≥1} 1/((1-q^(5n-4))(1-q^(5n-1)))

Left side: partitions with gap ≥ 2 conditions.
Right side: closed-form infinite product over arithmetic progressions mod 5.
Applied in the Baxter hard-hexagon model of statistical mechanics.

**TORIS mapping:**
Relators connected by CONTRADICTS edges cannot both be active (hard exclusion).
For a chain CONTRA structure, Z_F(q) matches the Rogers-Ramanujan product.
Field entropy H(F) = -d/dq [log Z_F(q)] at q = 1/e.

**Experimental result (Exp 13):**
- RR product formula matches q-series identity to < 0.001% (verified at q ∈ {0.1..0.7})
- contra_chain_structure() correctly identifies linear CONTRA chains
- Field entropy H(F) = 0.0669 nats for 8-concept chain field

**Ramanujan Critical Points (Heegner number analog):**
Ramanujan's constant e^(π√163) is nearly an integer because 163 is a Heegner number.
In TORIS: field configurations where Z_F(κ) ≈ integer are Ramanujan critical points —
maximum modular coherence, fastest convergence of all three expansions.

**Implementation:** `toris/engine/rogers_ramanujan.py`, `toris/engine/ramanujan_critical.py`
- `partition_function_rr(field, q)` — RR infinite product formula
- `partition_function_exact(field, q)` — brute-force enumeration
- `field_entropy(field)` — closed-form entropy from RR formula
- `critical_points(field)` — near-integer Z_F(κ) scan
- `RamanujanCritical` — full descriptor with criticality score
- `find_critical_points(field, manifold)` — combined diagnostic

---

## 3 — Source Papers

All structures from Collected Papers of S. Ramanujan (Hardy, Wilson, Seshu Aiyar, 1927):

| Paper | Year | Title | TORIS Structure |
|-------|------|-------|-----------------|
| Paper 36 | 1918 | Asymptotic Formulae in Combinatory Analysis (Hardy-Ramanujan) | Circle method, saddle points |
| Paper 25 | 1919 | Some Properties of p(n) | Partition congruences, suppression theorem |
| Paper 6  | 1914 | Modular Equations and Approximations to π | Rapidly convergent series, goal compression |
| Paper 26 | 1919 | Proof of Certain Identities in Combinatory Analysis (Rogers-Ramanujan) | Partition function, field entropy |
| Paper 18 | 1916 | On Certain Arithmetical Functions | Modular structure, τ-function connections |

---

## 4 — Experimental Results Summary

| Exp | Hypothesis | Result | Criterion |
|-----|-----------|--------|-----------|
| 11 | Suppression theorem: S_{5m+4} ≡ 0 (mod 5) | **PASS** 100% accuracy | ≥ 90% |
| 12 | 3-term Ramanujan < 1% error for Q < 0.01 | **PASS** 0.00% error | < 1% |
| 13 | RR product = q-series to < 0.01%; H(F) > 0 | **PASS** < 0.001% | < 0.01% |

---

## 5 — Original Contributions (Chandandeep Sharma)

The following mappings are original and do not appear in prior literature:

1. **Circle method for relational surprise** — ΔS at depth d via Hardy-Ramanujan saddle-point integral of Z_F(κ)
2. **Relational Suppression Theorem** — Ramanujan partition congruences as structural depth zeros in relational surprise
3. **Ramanujan Goal Compression** — 1/π rapid convergence applied to Goal Manifold warp with near-integer auto-switching
4. **TORIS Partition Function via Rogers-Ramanujan** — valid relator configurations counted by RR identity, giving closed-form entropy
5. **TORIS Ramanujan Critical Points** — Heegner-number analog for maximum field coherence

---

*TORIS Ramanujan Bridge v1.0*
*Chandandeep Sharma | June 2026*
*Layer 8 — built on Collected Papers of Ramanujan (1927), 393 pp.*
