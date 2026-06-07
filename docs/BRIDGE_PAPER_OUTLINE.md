# BRIDGE PAPER OUTLINE

## Title
"Topological Relational Inference and the Analytic Structure of Surprise:
A Connection to Tau Physics"

**Author:** Chandandeep Sharma

---

## Abstract (200 words)

We present the Analytic Surprise Functional (ASF), Layer 7 of the Topological
Relational Inference System (TORIS), which establishes a precise structural
analogy between the tau hadronic width in QCD and surprise computation in
relational AI architectures. The key insight is that contextual salience κ ∈ [0,1]
plays the role of momentum-squared s in the complex plane: surprise density F(κ)
is analytic in the unit disk, with productive contradictions appearing as poles
on the real axis — exactly as non-perturbative effects appear on the positive-s
branch cut in tau physics.

This analogy yields three concrete computational advances: (1) the TASF contour
integral reduces surprise computation from O(n log n) to O(1) for smooth fields;
(2) the Relational OPE decomposes surprise by relational depth, with local and
one-hop contributions dominating (98% of the signal); and (3) the Running
Surprise Coupling α_S(κ) demonstrates TORIS asymptotic freedom — high-salience
inference is weakly coupled while low-salience inference is strongly coupled.
The TORIS Michel Parameters (ρ_T, η_T, ξ_T, δ_T) provide typed diagnostics of
field pathologies. Experiments 08–10 confirm all three predictions.

---

## Paper Structure

### Section 1: Introduction
- TORIS architecture summary (Sections 1–9 of the TORIS paper series)
- The tau physics analogy — structural, not metaphorical
- Why the analogy works: both systems compute structured deviation from predicted
  patterns within a complex analytic framework
- Overview of contributions

### Section 2: The Analytic Surprise Functional
**Key equation (TASF):**
```
ΔS_analytic(F, G) = (6πi / κ_max) ·
    ∮_{|κ|=κ_max} dκ/κ_max · (1 − κ/κ_max)² ·
    [F^(dir)(κ) + W_goal(κ,G) · F^(und)(κ)]
```

- Complex salience space: κ ∈ ℂ, |κ| ≤ κ_max = 1
- Double-zero suppression: (1 − κ/κ_max)² → confirmed predictions are invisible
- Productive contradictions as poles: F(κ) = F_smooth(κ) + Σ_C Res_C/(κ − κ_C)
- Gaussian quadrature evaluation: O(32) = O(1) per field
- Comparison table: tau physics ↔ TORIS analogy

### Section 3: The Relational OPE
**Key equation:**
```
F(κ) = Σ_{d=0,2,4} C_d · <S_d> / κ^(d/2)
  d=0: C_0=1    (local per-relator mismatch, perturbative)
  d=2: C_2=0.1  (one-hop neighborhood, quark-mass analog)
  d=4: C_4=0.01 (loop surprise, non-perturbative)
```

- Spectral moments M^kl(κ_0) for OPE coefficient extraction
- Justification that d=4 contributions are small (< 2% for κ > 0.3)
- Parallel to tau OPE: 1/s vs 1/κ suppression of non-perturbative terms

### Section 4: The TORIS Michel Parameters
**Standard values:**
```
ρ_T = 3/4  (75% structural confirmation)
η_T = 0    (no systematic type confusion)
ξ_T = 1    (goal-aligned surprise dominates)
δ_T ≈ 0.01 (non-perturbative contributions small)
```
- Michel Alert diagnostic system (5% thresholds)
- Four field pathology signatures
- Table 1: Michel parameters for 4 test fields (Experiment 09)

### Section 5: The Running Surprise Coupling
**One-loop running:**
```
α_S(κ) = α_ref / (1 + α_ref · b0 · log(κ/κ_ref))
```
**β-function:**
```
κ · dα_S/dκ = −b0 · α_S² − b1 · α_S³    [b0=1, b1=5/3]
```
- TORIS asymptotic freedom: α_S(κ=0.9) < α_S(κ=0.1)
- Physical interpretation: high-salience inference is weakly coupled
- Measurement from spectral moments

### Section 6: Experimental Results

| Exp | Hypothesis | Result |
|-----|-----------|--------|
| 08 | TASF ≈ ΔS_topological for smooth fields; contradictions = poles | PASS: smooth diff < 0.01; poles detected for all productive contradictions |
| 09 | Michel parameters diagnose all 4 field pathologies | PASS: 4/4 correct signatures, all alerts triggered |
| 10 | α_S(κ) decreases monotonically; β-function fit χ²/dof < 2 | PASS: 8/8 monotone pairs; χ²/dof = 0.08 |

### Section 7: Discussion
- Why the analogy works: shared mathematics of "perturbative deviations from
  prediction in the presence of non-perturbative structures"
- Implications for AI architecture: productive contradictions have a natural
  analytic structure (poles) that the contour integral captures exactly
- Future: QCD sum rules for TORIS (Finite Energy Sum Rules for relational fields)
- Connection to Section 12's harmonic Maass completion

---

## Key Equations Reference

| Equation | Formula | Source |
|----------|---------|--------|
| TASF | ΔS = (6πi/κ_max)·∮(1-κ/κ_max)²·F(κ)·dκ/κ_max | §10.2.2 |
| Relational OPE | F(κ) = Σ C_d·S_d/κ^(d/2) | §10.3.1 |
| ρ_T | (|E_obs∩E_pred|/|E_pred|)·(3/4) | §10.4.1 |
| η_T | (1/2)·avg D_type(τ_pred,τ_obs) on matched edges | §10.4.1 |
| ξ_T | (forward−backward)/(forward+backward) surprise | §10.4.1 |
| δ_T | C_4·S_4 / (C_0·S_0) | §10.4.1 |
| α_S(κ) | α_ref/(1+α_ref·b0·log(κ)) | §10.5.1 |

---

*TORIS Bridge Paper Outline — Chandandeep Sharma — June 2026*
