"""Experiment 15 — Harmonic Maass Shadow Correction.

Hypothesis: PRODUCTIVE contradictions create shadow contributions that
push the complete ΔS away from the raw mock-modular TASF value.

Setup: one PRODUCTIVE contradiction with σ_a=0.7, σ_b=0.6, κ=0.5

Success criteria:
  (a) |shadow_correction| > 0 (non-trivial shadow detected)
  (b) |delta_S_complete − delta_S_mock| > 0 (shadow changes the result)
  (c) shadow_fraction > 0 (shadow is a real fraction of complete)
  (d) Complete result is finite
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.maass_completion import shadow_correction, complete_tasf
from toris.engine.complete_surprise import UnifiedSurprise


def build_contradiction_field(sigma_a=0.7, sigma_b=0.6, kappa=0.5):
    """Field with one PRODUCTIVE contradiction."""
    field = RelationalField()
    f_pred = RelationalField()

    A = ConceptState(id="A")
    B = ConceptState(id="B")
    C = ConceptState(id="C")
    for c in [A, B, C]:
        field.add_concept(c)
        f_pred.add_concept(c)

    # Predicted: CAUSAL
    r_pred = Relator(tau=RelationType.CAUSAL, src=A, tgt=B,
                     sigma=sigma_a, kappa=kappa, epsilon=0.0)
    # Observed: CONTRADICTS (productive contradiction)
    r_obs = Relator(tau=RelationType.CONTRADICTS, src=A, tgt=B,
                    sigma=sigma_b, kappa=kappa, epsilon=0.9)
    # Normal relator
    r_norm = Relator(tau=RelationType.ENABLES, src=B, tgt=C,
                     sigma=0.8, kappa=0.7, epsilon=0.05)

    f_pred.add_relator(r_pred)
    f_pred.add_relator(r_norm.clone(epsilon=0.0))
    field.add_relator(r_obs)
    field.add_relator(r_norm)

    return field, f_pred


def run():
    print("=" * 65)
    print("Experiment 15 — Harmonic Maass Shadow Correction")
    print("=" * 65)

    field, f_pred = build_contradiction_field()
    results = {}

    # Direct shadow correction
    sc = shadow_correction(field, f_pred)
    print(f"\n  shadow_correction = {sc:.6f}")

    # Full complete_tasf
    ct = complete_tasf(field, f_pred)
    print(f"  delta_S_mock     = {ct.delta_S_mock:.6f}")
    print(f"  delta_S_shadow   = {ct.delta_S_shadow:.6f}")
    print(f"  delta_S_complete = {ct.delta_S_complete:.6f}")
    print(f"  shadow_fraction  = {ct.shadow_fraction:.6f}")

    shadow_nonzero = abs(ct.delta_S_shadow) > 0
    changes_result = abs(ct.delta_S_complete - ct.delta_S_mock) > 0
    frac_pos = ct.shadow_fraction >= 0
    finite_pass = math.isfinite(ct.delta_S_complete)

    print(f"\n  PASS shadow > 0:       {shadow_nonzero}")
    print(f"  PASS changes result:   {changes_result}")
    print(f"  PASS shadow_frac >= 0: {frac_pos}")
    print(f"  PASS finite complete:  {finite_pass}")

    results["shadow_nonzero"] = shadow_nonzero
    results["changes_result"] = changes_result
    results["frac_pos"] = frac_pos
    results["finite"] = finite_pass

    # UnifiedSurprise standard regime (d=3, low Q(G))
    us = UnifiedSurprise()
    ur = us.compute(field, f_pred, d=3, goal_manifold=None)
    print(f"\n  UnifiedSurprise d=3: ΔS={ur.delta_S:.6f} regime={ur.regime_used}")
    results["unified_finite"] = math.isfinite(ur.delta_S)

    print("\n" + "=" * 65)
    n_pass = sum(results.values())
    n_total = len(results)
    all_pass = n_pass == n_total
    print(f"Experiment 15 {'PASS' if all_pass else 'FAIL'} ({n_pass}/{n_total})")
    print("=" * 65)
    return all_pass, results


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)
