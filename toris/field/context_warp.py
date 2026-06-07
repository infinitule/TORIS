"""Context-driven field warping (the TORIS spec §2, MATH_SPEC §3.3).

This module exposes the *context* axis of the warp operator Φ:
context ≠ goal.  A goal is a structured manifold (GoalManifold) that warps the
field by recomputing κ values against the goal stack.  Context is a lighter
signal — an ambient frame of reference (e.g. "emergency mode", "tutoring
register", "high-uncertainty regime") that scales salience multiplicatively.

Formally:

    Φ_context(ctx, F) = F' where:
        κ'(R) = clamp(κ(R) · ctx.weight(R.τ), 0, 1)

Context objects are thin wrappers that assign a multiplier to each
RelationType.  They compose left-to-right:

    Φ(ctx_b, Φ(ctx_a, F)) = Φ(ctx_a ⊕ ctx_b, F)   (approximate, order matters)

This is distinct from goal-warping (goal/warp.py) which uses the GoalManifold
and fires the contradiction scan.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Optional

from toris.field.relational_field import RelationalField
from toris.primitives.relation_types import RelationType


@dataclass
class Context:
    """An ambient context that scales relational salience by type.

    Attributes:
        name:    Human-readable label (e.g. "emergency", "tutorial").
        weights: RelationType → float multiplier ∈ (0, ∞).
                 Types absent from weights get multiplier 1.0 (identity).
        baseline: Fallback multiplier for unlisted types.  Default 1.0.
    """
    name: str = "default"
    weights: Dict[RelationType, float] = field(default_factory=dict)
    baseline: float = 1.0

    def weight(self, tau: RelationType) -> float:
        return self.weights.get(tau, self.baseline)

    def compose(self, other: "Context") -> "Context":
        """Return a new Context that applies self then other."""
        merged: Dict[RelationType, float] = {}
        all_types = set(self.weights) | set(other.weights)
        for tau in all_types:
            merged[tau] = self.weight(tau) * other.weight(tau)
        return Context(
            name=f"{self.name}⊗{other.name}",
            weights=merged,
            baseline=self.baseline * other.baseline,
        )


# ── Predefined contexts ──────────────────────────────────────────────────────

def emergency_context() -> Context:
    """Amplify CAUSAL, ENABLES, CONDITIONAL; suppress ANALOGOUS, REFINES."""
    return Context(
        name="emergency",
        weights={
            RelationType.CAUSAL: 2.0,
            RelationType.ENABLES: 1.8,
            RelationType.CONDITIONAL: 1.6,
            RelationType.ANALOGOUS: 0.3,
            RelationType.REFINES: 0.2,
        },
    )


def analogy_context() -> Context:
    """Amplify ANALOGOUS, INSTANTIATES, REFINES; suppress CAUSAL."""
    return Context(
        name="analogy",
        weights={
            RelationType.ANALOGOUS: 2.0,
            RelationType.INSTANTIATES: 1.7,
            RelationType.REFINES: 1.5,
            RelationType.CAUSAL: 0.4,
        },
    )


def high_uncertainty_context() -> Context:
    """Amplify CONTRADICTS and VIOLATES so tensions surface more readily."""
    return Context(
        name="high_uncertainty",
        weights={
            RelationType.CONTRADICTS: 2.0,
            RelationType.VIOLATES: 1.8,
            RelationType.EVIDENCES: 1.5,
        },
    )


# ── Core warp function ────────────────────────────────────────────────────────

def apply_context(ctx: Context, field: RelationalField) -> RelationalField:
    """Return a new RelationalField with κ values scaled by *ctx*.

    The input field is NOT mutated.  The returned field is a structural copy
    with adjusted salience values; strengths σ are unchanged.

    Algorithm (the TORIS spec §1.1 — not attention weighting):
      For each Relator R in F:
        κ' = clamp(κ · ctx.weight(R.τ), 0.0, 1.0)
      Return F' with updated κ values.
    """
    warped = RelationalField()
    # Re-register all concepts first
    for c in field.concepts():
        warped.add_concept(c)
    # Apply context scaling
    for r in field.relators():
        new_kappa = min(1.0, max(0.0, r.kappa * ctx.weight(r.tau)))
        warped.add_relator(r.clone(kappa=new_kappa))
    return warped


def compose_contexts(*contexts: Context) -> Context:
    """Left-fold composition of an arbitrary number of Context objects."""
    if not contexts:
        return Context(name="identity")
    result = contexts[0]
    for ctx in contexts[1:]:
        result = result.compose(ctx)
    return result
