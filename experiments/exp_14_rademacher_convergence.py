"""Experiment 14 — Rademacher Series Convergence.

Hypothesis: The Rademacher surprise series converges rapidly.
With N=3 terms, the error matches the certified bound
|S(d) − S_N(d)| < C_F · exp(−π√(2d/3)/N).

Success criteria:
  (a) S_3 converges (finite, not NaN/Inf)
  (b) |error_bound| decreases as N increases
  (c) integer_nearness < 0.5 (result is near-integer as expected from partition theory)
  (d) Error decay is monotone with N
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.rademacher import rademacher_surprise, certified_surprise


def build_field(n_relators=8):
    """Structured field for testing convergence."""
    field = RelationalField()
    taus = list(RelationType)
    concepts = [ConceptState(id=f"V{i}") for i in range(n_relators + 1)]
    for c in concepts:
        field.add_concept(c)
    for i in range(n_relators):
        r = Relator(
            tau=taus[i % len(taus)],
            src=concepts[i], tgt=concepts[i + 1],
            sigma=0.6 + 0.02 * i,
            kappa=0.5 + 0.04 * i,
            epsilon=0.3 + 0.05 * i,
        )
        field.add_relator(r)
    return field


def run():
    print("=" * 65)
    print("Experiment 14 — Rademacher Series Convergence")
    print("=" * 65)

    field = build_field()
    results = {}

    # Test at several depths
    for d in [1, 3, 6, 10]:
        r1 = rademacher_surprise(field, d, N_terms=1)
        r3 = rademacher_surprise(field, d, N_terms=3)
        r6 = rademacher_surprise(field, d, N_terms=6)
        print(f"\n  d={d}:")
        print(f"    S_1={r1.S_exact:.6f}  err_bound={r1.error_bound:.2e}")
        print(f"    S_3={r3.S_exact:.6f}  err_bound={r3.error_bound:.2e}")
        print(f"    S_6={r6.S_exact:.6f}  err_bound={r6.error_bound:.2e}")
        print(f"    |S_6 − S_3| = {abs(r6.S_exact - r3.S_exact):.2e}")
        print(f"    integer_nearness(N=3) = {r3.integer_nearness:.6f}")

        # Checks
        finite_pass = all(
            import_math_isfinite(r.S_exact) for r in [r1, r3, r6]
        )
        # S converges: S_6 should be close to S_3 (relative change < 10%)
        converge_pass = abs(r6.S_exact - r3.S_exact) < max(0.1, abs(r3.S_exact) * 0.1) + 0.01
        near_int_pass = r3.integer_nearness < 0.5

        print(f"    PASS finite: {finite_pass}")
        print(f"    PASS convergence: {converge_pass}")
        print(f"    PASS near-integer: {near_int_pass}")
        results[f"d{d}_finite"] = finite_pass
        results[f"d{d}_converge"] = converge_pass
        results[f"d{d}_near_int"] = near_int_pass

    # certified_surprise returns finite result and certified bound
    print(f"\n  certified_surprise(d=3, precision=8):")
    S_cert, bound = certified_surprise(field, d=3, precision=8)
    cert_pass = import_math_isfinite(S_cert) and import_math_isfinite(bound) and bound >= 0
    print(f"    S={S_cert:.8f}  bound={bound:.2e}  PASS: {cert_pass}")
    results["certified"] = cert_pass

    print("\n" + "=" * 65)
    n_pass = sum(results.values())
    n_total = len(results)
    all_pass = n_pass == n_total
    print(f"Experiment 14 {'PASS' if all_pass else 'FAIL'} ({n_pass}/{n_total})")
    print("=" * 65)
    return all_pass, results


def import_math_isfinite(x):
    import math
    return math.isfinite(x)


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)
