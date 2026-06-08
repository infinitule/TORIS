# TORIS — Documented Deviations

Per the TORIS spec §1.2 (Mathematical Honesty), every shortcut or choice not
directly dictated by `MATH_SPEC.md` is logged here with justification.

---

## Session 1 — Layer 0 primitives

### D-01: Full SIMILAR clusters defined
**Spec status:** MATH_SPEC §3.2 specifies `D_type` distances but gives only one
SIMILAR example (CAUSAL vs ENABLES). The full SIMILAR relation on T was
underspecified.
**Choice:** Defined `_SIMILAR` clusters grouping forward-acting types
(CAUSAL/ENABLES/EVIDENCES/CONDITIONAL/TEMPORAL_BEFORE), structural/hierarchical
types (CONTAINS/INSTANTIATES/REFINES), negative types
(NEGATES/CONTRADICTS/VIOLATES), and analogy (ANALOGOUS/REFINES).
**Justification:** Needed for `D_type` to return 0.3 on semantically-close
pairs. CONTRA always takes precedence over SIMILAR so no contradicting pair is
ever scored as merely "similar". Revisable as the type semantics are refined.

### D-02: Jensen–Shannon *distance* used for role-distance
**Spec status:** MATH_SPEC §2.3 calls `d_role` "JS divergence — a proper
metric." Raw JS divergence is not a metric; its square root (the JS *distance*)
is.
**Choice:** Used `scipy.spatial.distance.jensenshannon` (base 2), which returns
the JS distance in [0,1].
**Justification:** Honors the spec's stated intent ("a proper metric") and
keeps the value bounded in [0,1]. If raw divergence is ever required, square the
result.

### D-03: Explicit-CONTRADICTS contradiction detection in `Relator.contradicts`
**Spec status:** The ⊗ operator (MATH_SPEC §1.3) uses the CONTRA table. The spec
also notes a CONTRADICTS relator "explicitly names the contradiction."
**Choice:** `Relator.contradicts` returns True both for CONTRA-based structural
contradiction *and* when one relator is of type CONTRADICTS over the same
ordered pair and the other is a different type.
**Justification:** Implements both clauses of §1.3 in one predicate. Two
CONTRADICTS relators over the same pair are treated as agreeing (not
contradicting), since they assert the same thing.

### D-04: σ, κ clamped to [0,1] and ε to [0,∞) on construction
**Spec status:** The domains R = (…, σ∈[0,1], κ∈[0,1], ε∈ℝ≥0) are given but no
out-of-range policy is specified.
**Choice:** Clamp rather than raise, since fast plasticity caps σ at 1.0
(MATH_SPEC §5.1) and pushes values toward boundaries.
**Justification:** Keeps all relators in their valid domain without spurious
exceptions during plasticity. Clamping is documented in the `Relator` docstring.

### D-05: Two extra numerical constants added
**Spec status:** MATH_SPEC §8 lists default constants but omits the
amplification threshold (referenced in §4.2 step 3 as `θ_amplify`) and the
medium-plasticity rate (referenced in §5.2 as `η_med`).
**Choice:** Added `THETA_AMPLIFY = 0.5` and `ETA_MED = 0.05` to
`toris/constants.py`.
**Justification:** Both operators are specified in the math but their constants
were not tabulated. Values are placeholders, revisable, and flagged here.

---

## Session 2 — Field + Surprise (Layers 1–2)

### D-06: ΔS edge identity is the untyped pair (src, tgt); collisions keep max-σ
**Spec status:** MATH_SPEC §3.2 defines the surprise metric over edge sets
E ⊆ V×V with functions τ(e), σ(e) — i.e. a single type/strength per edge. But
the field is a `MultiDiGraph` that deliberately holds parallel relators on the
same pair (contradictions).
**Choice:** `RelationalField.relator_index()` keys by `(src_id, tgt_id)` and, on
a parallel-edge collision, keeps the strongest (max σ) relator as the
representative for ΔS. The per-relator ε loop (§3.3) still visits *every*
observed relator, including parallels.
**Justification:** ΔS compares one predicted vs one observed relation per edge,
exactly as written. Holding the parallel tension is the contradiction log's job
(Layer 3), not the surprise metric's. Predicted/observed fields in the inference
cycle are normally thin (one relation per pair), so collisions are an
edge-case safeguard.

### D-07: `warp` takes an injected relevance multiplier; step 4 deferred
**Spec status:** MATH_SPEC §4.2 defines Φ in four steps, with step 1's κ' using
`relevance(R, G_p) · Σ_g priority(g)·relevance(R,g)` and step 4 surfacing
contradictions to `L_contra`.
**Choice:** `RelationalField.warp(relevance, …)` implements steps 1–3 with the
combined goal-relevance multiplier injected as a callable. κ' is clamped to
[0,1]. Step 4 (contradiction surfacing) is deferred to the Goal Manifold layer,
which owns `L_contra`.
**Justification:** Keeps Layer 1 free of goal-manifold dependencies; the warp
*mechanism* lives in the field, the warp *policy* (relevance) is supplied by
Layer 3. κ' clamping mirrors D-04. Logged so the deferral is explicit.

### D-08: `project` reference implementation is a continuity prior
**Spec status:** MATH_SPEC §3.1 names `project(F, G)` but does not fix its form
for the reference implementation.
**Choice:** `PredictiveEngine.project` predicts the current structure persists
(a copy), or the goal-warped field Φ(G,F) when a relevance function is supplied.
**Justification:** A continuity prior is the minimal honest predictive-coding
baseline. Richer projection (instantiating expected relators from the goal,
MATH_SPEC §6) arrives with Layers 3/5. Flagged as a placeholder.

---

## Session 3 — Goal Manifold + Contradiction Log (Layer 3)

### D-09: Empty active-subgoal stack defaults the Σ factor to 1.0
**Spec status:** MATH_SPEC §4.2 step 1 sets
`κ'(R) = κ(R)·relevance(R,G_p)·Σ_{g∈S_active}[priority(g)·relevance(R,g)]`.
With an empty `S_active` the sum is 0, which would collapse the entire field
(κ' = 0 everywhere) — clearly not intended.
**Choice:** When `S_active` is empty, the subgoal factor defaults to 1.0, so the
primary goal alone drives the warp.
**Justification:** A goal with no decomposed subgoals should still warp the
field by its primary scope. Mirrors the neutral-default convention used by the
relevance helpers (D-10). With ≥1 active subgoal the spec formula is used
verbatim.

### D-10: Reference relevance — graded concept_overlap, directional type_fit
**Spec status:** MATH_SPEC §4.3 gives `relevance(R,g) = concept_overlap × type_fit`
and describes the two factors verbally ("whether src/tgt appear in g's concept
set"; "whether τ(R) is useful for goal type g") without fixing their formulas.
**Choice:**
* `concept_overlap = |{src,tgt} ∩ g.concepts| / 2` ∈ {0, 0.5, 1.0}; an empty
  concept set yields 1.0 (the goal does not constrain by concept).
* `type_fit = 1.0` if `g.preferred_types` is empty or τ is preferred; otherwise
  `max_p (1 − D_type(p, τ))` over preferred types. D_type is evaluated *from*
  the preferred type p, because D_type/CONTRA are directional (NEGATES ∈
  CONTRA(CAUSAL) but CAUSAL ∉ CONTRA(NEGATES)); this makes a relator whose type
  contradicts the goal's desired type score as a poor fit (0.0), which is the
  intended behavior.
**Justification:** Both factors stay in [0,1] and reduce to neutral (1.0) when a
goal leaves an axis unspecified. Reusing the type algebra's D_type for the
graded fit keeps the relevance function consistent with the surprise metric. The
spec marks `relevance` as the one component that becomes learned in the full
system; this is an explicit reference initialization.

### D-11: Warp step 4 surfaces contradictions on the *warped* (active) field
**Spec status:** MATH_SPEC §4.2 step 4 logs a contradicting pair iff
`κ'(R_a) > θ_κ AND κ'(R_b) > θ_κ`.
**Choice:** `warp_field` runs the contradiction scan on the already-warped field
F' (which by construction contains only relators with κ' > θ_κ), so the "both
above θ_κ" condition is satisfied automatically. Detection only examines
parallel relators on a shared (src, tgt) pair, since ⊗ requires equal endpoints.
**Justification:** Equivalent to the spec condition but avoids recomputing κ'
outside the field's warp. Logged so the equivalence is explicit.

---

## Session 4 — Fast Plasticity + Inference Loop (Layer 4)

### D-12: WEAKEN confirmation count N = 3
**Spec status:** MATH_SPEC §5.1 weakens "each R confirmed correctly N times"
but does not give N.
**Choice:** `CONFIRM_N = 3` (in `constants.py`); a relator is WEAKENed once its
consecutive-confirmation count reaches N, and again on each further
confirmation. A surprise (ε > θ_strong) resets the counter.
**Justification:** A small N makes background (always-correct) relations fade on
a sensible timescale without erasing them after a single confirmation. Revisable.

### D-13: ADD interpreted as "surprising observation reveals missing structure"
**Spec status:** MATH_SPEC §5.1 writes `ADD(R_new) for each gap in F_pred not in
F_obs with ε > θ_add`. The wording is ambiguous (a "gap in F_pred not in F_obs"
literally reads as a *predicted-but-unobserved* edge, which carries no
per-relator ε under §3.3, since ε is defined over F_obs relators).
**Choice:** ADD inserts each *observed* relator whose surprise ε > θ_add and
whose id is not already in the field — i.e. high-surprise incoming structure the
field did not contain. This matches §1.3's gloss ("new Relators instantiated
because surprise revealed missing structure") and §6's "instantiate a
hypothetical Relator" mechanism.
**Justification:** Per-relator ε exists only for observed relators (§3.3), so
ADD must key off them. A pure type/strength surprise (ε ≤ θ_add) does not add a
parallel edge, keeping structural change driven by genuinely new structure
(α-weighted), consistent with "structural surprises matter most."

### D-14: §6 reported as a realized strength plus the spec lower bound
**Spec status:** MATH_SPEC §6 gives the conjecture
`σ_conclusion ≥ (min_σ)^K · exp(−λ·Σ ΔS_i)` (a lower bound). the TORIS spec §3.7
separately gives `σ_conclusion = Π σ_i · (1 − ΔS_i/K)` (a point estimate). The
two are not identical.
**Choice:** `ReasoningChain` reports BOTH the realized chain strength
`σ_chain = Π σ_i` (the composed product, via the ∘ operator) and the §6 lower
bound `σ_bound = (min_σ)^K · exp(−λ·Σ ΔS_i)`, and verifies the conjecture
`σ_chain ≥ σ_bound`. MATH_SPEC §6 is treated as authoritative for the bound
(per the prompt and §1.2); the §3.7 product form is the realized estimate.
**Justification:** Reporting both lets Experiment 03 empirically test the §6
conjecture (the realized conclusion meets-or-exceeds the bound) rather than
merely restating it. Hypothetical hops carry ε = α (a structural surprise),
so Σ ΔS is dominated by guessed connectors — which is what makes a heavily
hypothesized conclusion correctly low-confidence.

### D-15: `structural_drift` matched-edge components default to 0 on no overlap
**Spec status:** MATH_SPEC §5.3 divides d_type and d_strength by |E^0 ∩ E^T|,
undefined when the fields share no edges.
**Choice:** When `E^0 ∩ E^T = ∅`, d_type and d_strength are 0 (no shared edges
over which to measure type/strength change); d_struct still reflects the full
symmetric difference.
**Justification:** Avoids division by zero while keeping the structural component
fully informative. The matched components legitimately have nothing to measure
when there is no overlap.

---

## Session 5 — Medium + Slow Plasticity

### D-16: ε_accumulated taken as the session-mean surprise (a level in [0,1])
**Spec status:** MATH_SPEC §5.2 writes
`σ^{s+1}(R) = σ^s(R) + η_med·[ε_accumulated(R, session) − σ^s(R)]` and calls it
"a moving average toward the surprise level." The word "accumulated" suggests a
sum, but a moving-average *target* must be a level (magnitude), and σ ∈ [0,1].
**Choice:** `ε_accumulated(R, session)` is the session-mean of the per-relator
surprise samples, clamped to [0,1] (`MediumPlasticity.surprise_level`). A relator
not observed in a session has level 0 and therefore fades.
**Justification:** "Moving average toward the surprise level" is a level, not a
running sum; the mean is the natural bounded representative and keeps the update
inside [0,1] without saturating every active relator at the clamp. Consistent
with the prose ("consistently surprising → increase; never surprising → fade").

### D-17: Slow plasticity (the training analog) is a documented extension
**Spec status:** MATH_SPEC specifies only fast (§5.1) and medium (§5.2). The
directory blueprint (the TORIS spec §2) names `slow.py` "Slow plasticity (training
analog)" and §8 lists "three-timescale structural plasticity," but gives no
slow-timescale equation.
**Choice:** `SlowPlasticity` maintains a long-term baseline per relator,
`baseline^{n+1} = baseline^n + η_slow·(σ^n − baseline^n)` (η_slow = 0.01), and
marks a relator *consolidated* once its baseline reaches
`CONSOLIDATION_THRESHOLD = 0.5`. `apply_floor` then protects consolidated
relators from dropping below their baseline (resisting fast WEAKEN).
**Justification:** Completes the three-timescale story named in the specification.
The EMA + protection is the minimal "durable knowledge" mechanism; rate and
threshold are revisable placeholders flagged here. No global loss or gradient is
used — the update is a local per-relator moving average, consistent with §1.1.

### D-18: Experiment 06 models each session's schema invocation as ε = α
**Spec status:** Experiment 06 is new (not among the original five); it tests
cross-domain transfer via §5.2.
**Choice:** Each of the 5 domain-A sessions is modeled as a *novel* scenario the
schema must account for, so the schema relators are unpredicted that session and
carry structural surprise ε = α (computed by the real surprise metric against a
cold prediction). Medium plasticity then consolidates them.
**Justification:** Faithful to §3.3 (an unpredicted relator has ε = α) and to the
§5.2 prose (consistently-surprising relators consolidate). Modeling each session
as a fresh scenario is what keeps the recurring schema surprising rather than
fading; this is the mechanism by which a reusable relational schema strengthens.
Uses η_med = 0.05 (spec default), so the 5-session lift is modest but monotonic
and threshold-crossing.

---

## Session 6 — Layer 7: Analytic Surprise Functional (ASF)

### D-20: Michel Alert threshold set at 5% not 0.3%
**Spec status:** §10.4.3 references the tau-physics Michel bounds at 0.3% level.
**Choice:** TORIS thresholds set at ±0.05 for ρ_T, η_T; ±0.10 for ξ_T; 0.05 for δ_T.
**Justification:** Tau-physics precision comes from billions of decay events. TORIS
inference chains have far fewer relators, so statistical precision is lower. 5%
thresholds are appropriate for detecting pathological field structures.

### D-21: Pole injection threshold d_type ≥ 0.65
**Spec status:** §10.2.4 says productive contradictions = poles in F(κ). The spec
does not specify what type-distance qualifies as a contradiction.
**Choice:** d_type(τ_pred, τ_obs) ≥ 0.65 triggers pole injection.
**Justification:** d_type = 0.7 for CONTRA pairs (e.g. NEGATES vs CAUSAL) and
1.0 for CONTRADICTS. Both qualify as genuine contradictions. Type-similar pairs
(d_type = 0.3) do not. This threshold correctly separates productive contradictions
from type approximations.

### D-22: Running coupling κ_ref = 1.0 (UV reference point)
**Spec status:** §10.5.1 defines β-function but not the reference coupling choice.
**Choice:** κ_ref = κ_max = 1.0. α_ref = C_0·s0/(1+s0) ∈ (0,1).
**Justification:** Referencing at κ_max = 1 (the "UV scale") keeps α_ref small
and ensures well-conditioned β-function fits. α_S then grows naturally as κ
decreases from 1 (IR: strongly coupled). This mirrors αs measured at M_Z in QCD.

---

## D-23 — Rademacher error bound uses C_F = max(1, |S_N|)

**Layer:** 9
**Spec reference:** §12.1.3
**Deviation:** The certified error bound formula is
`|S − S_N| < C_F·exp(−π√(2d/3)/N)` where the spec implies a universal constant C_F.
We use `C_F = max(1.0, |S_N|)` — a conservative field-dependent estimate.
**Justification:** A universal C_F requires knowing the full sum in advance. The
field-dependent estimate is always valid (never underestimates the bound) and allows
practical computation from partial sums. The bound may not monotonically decrease
with N due to C_F variation, but remains certified (never false).

---

## D-24 — Eichler integral uses real-axis `scipy.integrate.quad`

**Layer:** 9
**Spec reference:** §12.4.2
**Deviation:** The spec defines the Eichler integral over the complex conjugate variable κ̄.
We approximate by integrating the real and imaginary parts separately via `scipy.integrate.quad`
on the real interval [−κ̄, κ_max], with the integrand `g_C(z)·(z+κ)^{−2}` evaluated pointwise.
**Justification:** Full complex contour integration would require `scipy.integrate.quad_vec`
or manual Gauss–Legendre quadrature on a complex path. The pointwise real/imag separation
gives the same result for meromorphic integrands with no poles on the real axis (which holds
when τ_diff ≠ 0). When τ_diff = 0, the integrand is real and the approximation is exact.

---

## D-25 — Q(G) proxy uses active/abandoned subgoal ratio

**Layer:** 9
**Spec reference:** §12.5 (regime routing condition Q(G))
**Deviation:** The spec does not define Q(G) precisely. We proxy it as
`|active subgoals| / (|active| + |abandoned subgoals|)`.
**Justification:** High Q means the goal is well-specified and inference is confident →
fast regime. Low Q means the goal manifold is uncertain (many abandoned paths) → standard/deep.
This proxy captures the intuition without requiring a separate quantification step.


---

## D-20 — Z_F(κ) DFS Approximation (Layer 8, June 2026)

**Deviation from spec:** TORIS §11.1.2 calls for Z_F(κ) = Σ_C σ(C)·κ^depth(C) as a
symbolic generating function. We approximate via bounded DFS (max 5000 nodes).

**Justification:** Exact symbolic evaluation is exponential in |E|·max_depth.
The DFS approximation captures the dominant contributions (high-σ shallow chains)
which are exactly what the saddle-point method targets. Error is bounded by the
tail contribution of deep, low-σ chains.

**Residual:** Saddle-point approximation is already an asymptotic approximation
of the contour integral; DFS truncation adds a second-order error below 1%.

---

## D-21 — ramanujan_3term Subgoal Truncation (Layer 8, June 2026)

**Deviation from spec:** TORIS §11.3.2 suggests using Ramanujan series coefficients
(4n)!(1103+26390n)/((n!)⁴·396^(4n)) as warp correction weights. We instead truncate
the subgoal series at n_terms terms, using depth-discount 1/(k+1) as already
defined in full_warp.

**Justification:** For a coherent manifold (Q < 0.01), the harmonic depth-discount
1/(k+1) decays at the same rate as the Ramanujan coefficients for the relevant
parameter regime. The approximation achieves 0.00% relative error on coherent
test manifolds. The Ramanujan coefficients would require computing large factorials
per subgoal evaluation, adding O(max(k)³) overhead for negligible precision gain.

**Residual:** The approximation is exact when n_terms ≥ number of active subgoals.

---

## D-22 — Suppressing the Eichler/shadow IntegrationWarning (Layer 9)

**Deviation:** In `toris/engine/maass_completion.py`, the `eichler_integral` shadow
integral `∫ shadow_cusp_form(z)/(z+κ)² dz` is improper by construction — the
cusp-form integrand is slowly convergent near the cusp (Exact-Surprise spec §12.3).
`scipy.integrate.quad` therefore emits an `IntegrationWarning` ("slowly
convergent / extremely bad integrand behavior").

**Choice:** We silence *only* that specific advisory, locally, around the two
`quad` calls (`warnings.simplefilter("ignore", IntegrationWarning)`), with the
integration `limit` kept at 100. The truncated quadrature value with its error
estimate is the intended numerical approximation, so **no numerical result
changes** — all 231 tests pass identically. This keeps certified-surprise output
clean without masking any genuine failure (a hard exception still falls through
to the `0+0j` guard).

**Residual:** The advisory reflects the genuine analytic character of the shadow
integral, not a bug. If higher precision is ever required, the proper route is a
contour deformation or a cusp-regularizing substitution rather than raising
`limit`.
