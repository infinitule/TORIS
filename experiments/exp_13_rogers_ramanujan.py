"""Exp 13 — Rogers-Ramanujan Partition Function (TORIS §11.4 / §11.6).

The Rogers-Ramanujan identity (first):
    1 + Σ_{n≥1} q^(n²)/((1-q)…(1-qⁿ))  =  Π_{n≥1} 1/((1-q^(5n-4))(1-q^(5n-1)))

Left (q-series) = Right (infinite product).

Tests:
  1. Product formula matches q-series to < 0.01% (mathematical identity check).
  2. contra_chain_structure() correctly identifies chain-CONTRA fields.
  3. Field entropy H(F) > 0 (field has non-trivial reasoning flexibility).
  4. critical_points() finds near-integer Z(κ) configurations.
  5. RamanujanCritical report runs without error.

Success criterion:
  Product-vs-series error < 0.01% AND H(F) > 0 AND chain structure detected.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import math
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.field.relational_field import RelationalField
from toris.goal.manifold import GoalManifold, Goal
from toris.engine.rogers_ramanujan import (
    contra_chain_structure, partition_function_rr, partition_function_exact,
    field_entropy, critical_points,
)
from toris.engine.ramanujan_critical import find_critical_points, critical_report


def build_chain_contra_field(n: int = 8) -> RelationalField:
    """Build a field whose CONTRA graph is a simple linear chain X0-X1-...-Xn."""
    field = RelationalField()
    concepts = [ConceptState(id=f"X{i}") for i in range(n + 1)]
    for c in concepts:
        field.add_concept(c)
    # CAUSAL backbone (does not affect CONTRA graph)
    for i in range(n):
        field.add_relator(Relator(
            RelationType.CAUSAL, concepts[i], concepts[i + 1],
            sigma=0.8, kappa=0.7, epsilon=0.1,
        ))
    # CONTRA chain: X0-X1-X2-...-Xn (linear path = valid chain structure)
    for i in range(n):
        field.add_relator(Relator(
            RelationType.CONTRADICTS, concepts[i], concepts[i + 1],
            sigma=0.6, kappa=0.5, epsilon=0.0,
        ))
    return field


def rr_q_series(q: float, n_terms: int = 20) -> float:
    """Left-hand side of first Rogers-Ramanujan identity."""
    def q_pochhammer(q, n):
        p = 1.0
        for k in range(1, n + 1):
            p *= (1.0 - q ** k)
        return p
    S = 1.0
    for n in range(1, n_terms + 1):
        S += q ** (n * n) / q_pochhammer(q, n)
    return S


def run():
    print("=== Exp 13: Rogers-Ramanujan Partition Function ===\n")

    # 1. Mathematical identity check (independent of TORIS field)
    print("--- 1. Rogers-Ramanujan Identity Verification ---")
    test_qs = [0.1, 0.2, 0.3, 0.5, 0.7]
    max_rr_err = 0.0
    for q in test_qs:
        z_product = partition_function_rr(None, q, n_terms=50)  # type: ignore
        z_series = rr_q_series(q, n_terms=25)
        rel_err = abs(z_product - z_series) / max(abs(z_series), 1e-9)
        max_rr_err = max(max_rr_err, rel_err)
        print(f"  q={q:.1f}  product={z_product:.8f}  series={z_series:.8f}  "
              f"err={rel_err:.2e}")

    identity_ok = max_rr_err < 1e-4
    print(f"Identity match (< 0.01%): {'PASS' if identity_ok else 'FAIL'}  "
          f"max_err={max_rr_err:.2e}")

    # 2. Chain structure detection
    print("\n--- 2. Chain CONTRA Structure Detection ---")
    field = build_chain_contra_field(8)
    is_chain = contra_chain_structure(field)
    print(f"  contra_chain_structure: {is_chain}")
    chain_ok = is_chain

    # 3. Field entropy
    print("\n--- 3. Field Entropy H(F) ---")
    for q_val in [1.0 / math.e, 0.3, 0.5]:
        h = field_entropy(field, q=q_val)
        print(f"  H(F, q={q_val:.4f}) = {h:.6f} nats")
    h_default = field_entropy(field)
    entropy_ok = h_default > 0.0

    # 4. Critical points
    print("\n--- 4. Ramanujan Critical Points ---")
    cps = critical_points(field, steps=200, threshold=1e-2)
    print(f"  Critical points found: {len(cps)}")
    for cp in cps[:3]:
        print(f"    κ={cp.kappa:.4f}  Z={cp.Z_value:.6f}  frac={cp.fractional_part:.4f}")

    # 5. Full critical report
    print("\n--- 5. RamanujanCritical Report ---")
    primary = Goal(description="explore_field_entropy")
    manifold = GoalManifold(primary=primary)
    print(critical_report(field, manifold, threshold=1e-2))

    success = identity_ok and chain_ok and entropy_ok
    print(f"\nIdentity verified: {'PASS' if identity_ok else 'FAIL'}")
    print(f"Chain structure:   {'PASS' if chain_ok else 'FAIL'}")
    print(f"Entropy H(F) > 0:  {'PASS' if entropy_ok else 'FAIL'} ({h_default:.4f})")
    print(f"\n{'SUCCESS' if success else 'FAILURE'}: Exp 13 {'PASS' if success else 'FAIL'}")
    return success


if __name__ == "__main__":
    run()
