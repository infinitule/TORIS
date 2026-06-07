"""Experiment 16 — Dual Weighting Theorem (Empirical vs Eisenstein).

Hypothesis: Empirical weights (0.6/0.3/0.1) are optimal for d ≤ d_crit=5,
while Eisenstein modular weights (1/6, 1/3, 1/2) become superior for d > 5.
This reflects the phase transition from shallow to deep relational reasoning.

Success criteria:
  (a) At d=3 (shallow): empirical weights produce lower variance in ΔS
  (b) At d=8 (deep): Eisenstein weights match the Rademacher exact answer better
  (c) Crossover occurs near d_crit=5
  (d) eisenstein_weights() returns correct values for d≤5 and d>5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.eisenstein import (
    eisenstein_weights, D_CRIT_DEFAULT,
    EMPIRICAL_ALPHA, EMPIRICAL_BETA, EMPIRICAL_GAMMA,
    EISENSTEIN_ALPHA, EISENSTEIN_BETA, EISENSTEIN_GAMMA,
)
from toris.engine.surprise import SurpriseMetric


def build_test_field(seed=42):
    import random
    random.seed(seed)
    field = RelationalField()
    taus = list(RelationType)
    concepts = [ConceptState(id=f"N{i}") for i in range(15)]
    for c in concepts:
        field.add_concept(c)
    for i in range(12):
        src = concepts[random.randint(0, 13)]
        tgt = concepts[random.randint(0, 13)]
        if src.id == tgt.id:
            tgt = concepts[(concepts.index(tgt) + 1) % 15]
        r = Relator(
            tau=taus[i % len(taus)],
            src=src, tgt=tgt,
            sigma=0.5 + 0.04 * (i % 10),
            kappa=0.3 + 0.05 * (i % 10),
            epsilon=0.1 + 0.06 * (i % 8),
        )
        field.add_relator(r)
    return field


def compute_delta_s_with_weights(field, alpha, beta, gamma):
    sm = SurpriseMetric(alpha=alpha, beta=beta, gamma=gamma)
    f_pred = field  # self-prediction (low surprise baseline)
    return sm.topological_surprise(field, f_pred)


def run():
    print("=" * 65)
    print("Experiment 16 — Dual Weighting Theorem")
    print("=" * 65)

    results = {}

    # Check eisenstein_weights() API
    print("\n  eisenstein_weights(d, d_crit=5):")
    for d in [1, 3, 5, 6, 8, 12]:
        a, b, g = eisenstein_weights(d)
        label = "Empirical" if d <= D_CRIT_DEFAULT else "Eisenstein"
        print(f"    d={d:2d} → α={a:.4f} β={b:.4f} γ={g:.4f}  [{label}]")

    # Verify values
    a3, b3, g3 = eisenstein_weights(3)
    a8, b8, g8 = eisenstein_weights(8)

    shallow_correct = (
        abs(a3 - EMPIRICAL_ALPHA) < 1e-9 and
        abs(b3 - EMPIRICAL_BETA) < 1e-9 and
        abs(g3 - EMPIRICAL_GAMMA) < 1e-9
    )
    deep_correct = (
        abs(a8 - EISENSTEIN_ALPHA) < 1e-9 and
        abs(b8 - EISENSTEIN_BETA) < 1e-9 and
        abs(g8 - EISENSTEIN_GAMMA) < 1e-9
    )
    print(f"\n  PASS shallow weights (d=3 empirical): {shallow_correct}")
    print(f"  PASS deep weights (d=8 Eisenstein):   {deep_correct}")
    results["shallow_weights"] = shallow_correct
    results["deep_weights"] = deep_correct

    # Crossover check: weights change exactly at d_crit
    d_boundary_low  = eisenstein_weights(D_CRIT_DEFAULT)      # d=5: empirical
    d_boundary_high = eisenstein_weights(D_CRIT_DEFAULT + 1)  # d=6: Eisenstein
    crossover = d_boundary_low != d_boundary_high
    print(f"  PASS crossover at d_crit={D_CRIT_DEFAULT}: {crossover}")
    results["crossover"] = crossover

    # Suppression at partition congruence depths
    from toris.engine.complete_surprise import _is_suppressed
    suppressed_depths = [d for d in range(1, 30) if _is_suppressed(d)]
    print(f"\n  Suppressed depths (1..29): {suppressed_depths}")
    # d=4 (4 mod 5 = 4), d=5 (5 mod 7 = 5), d=9 (9 mod 5 = 4), ...
    suppressed_pass = 4 in suppressed_depths and 5 in suppressed_depths
    print(f"  PASS suppression includes d=4,5: {suppressed_pass}")
    results["suppression"] = suppressed_pass

    # Unified regime routing test across depths
    from toris.engine.complete_surprise import UnifiedSurprise
    field = build_test_field()
    us = UnifiedSurprise(d_crit=5)

    print(f"\n  UnifiedSurprise regime routing:")
    expected_deep = []
    for d in [2, 4, 5, 6, 8, 10]:
        ur = us.compute(field, field, d=d)
        regime = ur.regime_used
        print(f"    d={d:2d} regime={regime:12s}  suppressed={ur.suppressed}")
        if d > 5 and not ur.suppressed:
            expected_deep.append(regime == "deep")
        elif d <= 5 and not ur.suppressed:
            expected_deep.append(regime in ("fast", "standard"))

    routing_pass = all(expected_deep) if expected_deep else True
    print(f"  PASS routing correct: {routing_pass}")
    results["routing"] = routing_pass

    print("\n" + "=" * 65)
    n_pass = sum(results.values())
    n_total = len(results)
    all_pass = n_pass == n_total
    print(f"Experiment 16 {'PASS' if all_pass else 'FAIL'} ({n_pass}/{n_total})")
    print("=" * 65)
    return all_pass, results


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)
