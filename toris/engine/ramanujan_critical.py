"""TORIS Ramanujan Critical Points — field configurations of maximum modular
coherence, analogous to Heegner numbers (TORIS §11.5).

Ramanujan's constant: e^(π√163) ≈ 262537412640768743.999999999999250...
This near-integer arises because 163 is a Heegner number (class number 1).

In TORIS: a field is at a Ramanujan critical point when its partition function
Z_F(κ) is nearly an integer.  At these points:
  1. Modular coherence is maximum
  2. Ramanujan goal expansion converges fastest (3 terms sufficient)
  3. Circle method saddle-point approximation is most accurate
  4. Suppression theorem applies with maximum force

This module provides the unified diagnostic: given field + goal,
find and characterise all Ramanujan critical configurations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from toris.engine.rogers_ramanujan import (
    CriticalPoint,
    critical_points,
    field_entropy,
    partition_function_rr,
)
from toris.engine.ramanujan_goal import goal_coherence, near_integer_check
from toris.engine.suppression import is_modular_field, RAMANUJAN_RESIDUES

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold


@dataclass
class RamanujanCritical:
    """Full descriptor of a Ramanujan critical configuration."""

    kappa: float              # κ value where criticality was detected
    Z_value: float            # Z_F(κ) at this point
    coherence_Q: float        # goal coherence Q(G) — smaller is better
    is_near_integer: bool     # |Z − round(Z)| < threshold
    field_entropy: float      # H(F) at q = 1/e
    # Modular structure flags
    is_5_modular: bool
    is_7_modular: bool
    is_11_modular: bool
    # Recommendations
    use_ramanujan_expansion: bool   # Q < 0.01 → use 3-term expansion
    # Heegner analogy score ∈ [0,1]: 1 = all conditions met
    criticality_score: float


def _criticality_score(rc: "RamanujanCritical") -> float:
    """Aggregate 0-1 score: how deeply critical is this configuration?"""
    signals = [
        rc.is_near_integer,
        rc.coherence_Q < 0.01,
        rc.is_5_modular,
        rc.is_7_modular or rc.is_11_modular,
    ]
    return sum(signals) / len(signals)


def find_critical_points(
    field: "RelationalField",
    manifold: "GoalManifold",
    kappa_range=(0.05, 0.95),
    steps: int = 200,
    threshold: float = 1e-4,
) -> List[RamanujanCritical]:
    """Scan κ range and return all Ramanujan critical configurations.

    For each near-integer point of Z_F(κ), build a RamanujanCritical
    descriptor combining circle-method, suppression, and goal-coherence
    information.
    """
    cps = critical_points(field, kappa_range, steps, threshold)
    Q = goal_coherence(manifold)
    H = field_entropy(field)
    is5 = is_modular_field(field, 5)
    is7 = is_modular_field(field, 7)
    is11 = is_modular_field(field, 11)

    results: List[RamanujanCritical] = []
    for cp in cps:
        rc = RamanujanCritical(
            kappa=cp.kappa,
            Z_value=cp.Z_value,
            coherence_Q=Q,
            is_near_integer=True,
            field_entropy=H,
            is_5_modular=is5,
            is_7_modular=is7,
            is_11_modular=is11,
            use_ramanujan_expansion=(Q < 0.01),
            criticality_score=0.0,
        )
        rc.criticality_score = _criticality_score(rc)
        results.append(rc)
    return results


def is_at_critical(
    field: "RelationalField",
    manifold: "GoalManifold",
    threshold: float = 1e-4,
) -> bool:
    """Quick check: is the field currently at any Ramanujan critical point?"""
    return bool(find_critical_points(field, manifold, threshold=threshold))


def critical_report(
    field: "RelationalField",
    manifold: "GoalManifold",
    threshold: float = 1e-4,
) -> str:
    """Human-readable diagnostic report for Ramanujan criticality."""
    cps = find_critical_points(field, manifold, threshold=threshold)
    Q = goal_coherence(manifold)
    H = field_entropy(field)
    is5 = is_modular_field(field, 5)

    lines = [
        "=== Ramanujan Critical Point Report ===",
        f"Goal coherence Q(G)      : {Q:.6f}",
        f"Field entropy H(F)       : {H:.4f} nats",
        f"5-modular field          : {is5}",
        f"Critical points found    : {len(cps)}",
    ]
    for i, rc in enumerate(cps):
        lines.append(
            f"  [{i}] κ={rc.kappa:.4f}  Z={rc.Z_value:.6f}"
            f"  near_int={rc.is_near_integer}"
            f"  score={rc.criticality_score:.2f}"
        )
    if not cps:
        lines.append("  (no critical points in scanned range)")
    return "\n".join(lines)
