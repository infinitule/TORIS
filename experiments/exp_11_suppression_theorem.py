"""Exp 11 — Ramanujan Suppression Theorem Verification (TORIS §11.2 / §11.6).

Hypothesis:
  For fields with p-modular strength structure, S_{pm+r₀}(F) ≡ 0 (mod p)
  for the Ramanujan residues p=5/r₀=4, p=7/r₀=5, p=11/r₀=6.

Setup:
  - 50-relator modular field (strengths discretised to multiples of 1/5)
  - Compute S_d for d = 1..30
  - Assert S_{5m+4} ≡ 0 (mod 5) for all m
  - Compare with non-modular field (random strengths — no suppression expected)

Success criterion:
  ≥ 90% of depths d ≡ 4 (mod 5) are suppressed in modular field.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import random
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.field.relational_field import RelationalField
from toris.engine.suppression import (
    is_modular_field, suppression_report, suppressed_depth,
)

random.seed(42)


def build_modular_field(n_relators: int = 50, p: int = 5) -> RelationalField:
    """Build a field with strengths discretised to multiples of 1/p."""
    field = RelationalField()
    concepts = [ConceptState(id=f"M{i}") for i in range(15)]
    for c in concepts:
        field.add_concept(c)
    taus = [RelationType.CAUSAL, RelationType.ENABLES, RelationType.EVIDENCES,
            RelationType.CONTAINS, RelationType.REFINES]
    for i in range(n_relators):
        src, tgt = random.sample(concepts, 2)
        # strength is a multiple of 1/p → modular
        k = random.randint(0, p)
        sigma = k / p
        epsilon = random.randint(0, p) / p
        r = Relator(
            tau=random.choice(taus),
            src=src, tgt=tgt,
            sigma=sigma, kappa=0.5,
            epsilon=epsilon,
        )
        field.add_relator(r)
    return field


def build_random_field(n_relators: int = 50) -> RelationalField:
    """Non-modular field with unconstrained random strengths."""
    field = RelationalField()
    concepts = [ConceptState(id=f"R{i}") for i in range(15)]
    for c in concepts:
        field.add_concept(c)
    taus = [RelationType.CAUSAL, RelationType.ENABLES, RelationType.EVIDENCES]
    for i in range(n_relators):
        src, tgt = random.sample(concepts, 2)
        r = Relator(
            tau=random.choice(taus),
            src=src, tgt=tgt,
            sigma=random.random(),
            kappa=0.5,
            epsilon=random.random(),
        )
        field.add_relator(r)
    return field


def run():
    print("=== Exp 11: Ramanujan Suppression Theorem ===\n")

    mod_field = build_modular_field(50)
    rnd_field = build_random_field(50)

    is_mod = is_modular_field(mod_field, 5)
    is_mod_rnd = is_modular_field(rnd_field, 5)
    print(f"Modular field is_5_modular: {is_mod}")
    print(f"Random field is_5_modular:  {is_mod_rnd}")

    print("\n--- Modular field suppression (d=1..30) ---")
    report = suppression_report(mod_field, max_depth=30)

    print(f"{'d':>4}  {'S_d':>8}  {'expected_sup':>13}  {'actual_sup':>11}  {'mod_p':>6}")
    for e in report.entries:
        if e.expected_suppressed:
            print(f"{e.depth:>4}  {e.s_d:>8.4f}  {str(e.expected_suppressed):>13}  "
                  f"{str(e.actually_suppressed):>11}  {str(e.s_d_mod_p):>6}  "
                  f"(p={e.prime})")

    print(f"\nSuppression accuracy : {report.suppression_accuracy:.1%}")
    print(f"Correct predictions  : {report.n_suppressed_correct}")
    print(f"Wrong predictions    : {report.n_suppressed_wrong}")

    print("\n--- Random field (should show no reliable suppression) ---")
    rnd_report = suppression_report(rnd_field, max_depth=30)
    print(f"Suppression accuracy : {rnd_report.suppression_accuracy:.1%}")

    success = report.suppression_accuracy >= 0.9
    print(f"\n{'SUCCESS' if success else 'FAILURE'}: Exp 11 "
          f"({'PASS' if success else 'FAIL'} — accuracy={report.suppression_accuracy:.1%})")
    return success


if __name__ == "__main__":
    run()
