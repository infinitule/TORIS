"""TORIS typed relation set T and the algebra over it.

Implements MATH_SPEC priority item #1:
  - the ``RelationType`` enum (the typed relation set T)
  - symmetry declarations
  - the D_type semantic-distance matrix (MATH_SPEC §3.2)
  - the CONTRA contradiction table (MATH_SPEC §1.3)
  - the Ω composition-compatibility table (MATH_SPEC §1.2)

This module is pure: it defines the *types* and the relations *between* types.
It knows nothing about ConceptStates or fields. Higher layers consume it.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Dict, FrozenSet, Optional, Tuple

from toris.constants import (
    D_TYPE_CONTRA,
    D_TYPE_SAME,
    D_TYPE_SIMILAR,
    D_TYPE_UNRELATED,
)


class RelationType(Enum):
    """The typed relation set T (the TORIS spec §3.1, initial and extensible).

    Relators carrying these types are *asymmetric by default*. Only the
    types in ``SYMMETRIC_TYPES`` may be reversed without changing meaning.
    """

    CAUSAL = "CAUSAL"  # A produces B
    CONDITIONAL = "CONDITIONAL"  # A activates B under condition C
    CONTRADICTS = "CONTRADICTS"  # A and B cannot both hold simultaneously
    CONTAINS = "CONTAINS"  # A structurally includes B
    ENABLES = "ENABLES"  # A makes B possible but does not cause it
    VIOLATES = "VIOLATES"  # A is inconsistent with rule B
    ANALOGOUS = "ANALOGOUS"  # A and B share relational structure across domains
    REFINES = "REFINES"  # B is a more precise version of A
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"  # A occurs before B
    EVIDENCES = "EVIDENCES"  # A raises the probability of B
    NEGATES = "NEGATES"  # A suppresses B
    INSTANTIATES = "INSTANTIATES"  # A is a specific case of B

    def __repr__(self) -> str:  # concise, stable repr for logs and tests
        return f"RelationType.{self.name}"


# ---------------------------------------------------------------------------
# Symmetry declarations (the TORIS spec §3.1: "Symmetry must be explicitly declared")
# ---------------------------------------------------------------------------

#: Types whose meaning is unchanged when source and target are swapped.
SYMMETRIC_TYPES: FrozenSet[RelationType] = frozenset(
    {RelationType.CONTRADICTS, RelationType.ANALOGOUS}
)


def is_symmetric(tau: RelationType) -> bool:
    """Return True iff a relation of type ``tau`` is symmetric (A↔B)."""
    return tau in SYMMETRIC_TYPES


# ---------------------------------------------------------------------------
# CONTRA table (MATH_SPEC §1.3): implicit structural contradictions on T.
# CONTRA(τ) = the set of types that structurally contradict τ.
# ---------------------------------------------------------------------------

_CONTRA: Dict[RelationType, FrozenSet[RelationType]] = {
    RelationType.CAUSAL: frozenset({RelationType.NEGATES, RelationType.CONTRADICTS}),
    RelationType.ENABLES: frozenset({RelationType.NEGATES}),
    RelationType.EVIDENCES: frozenset({RelationType.NEGATES}),
    RelationType.CONTAINS: frozenset({RelationType.VIOLATES}),
    RelationType.VIOLATES: frozenset({RelationType.CONTAINS, RelationType.ENABLES}),
    RelationType.CONDITIONAL: frozenset({RelationType.CONTRADICTS}),
}


def contra(tau: RelationType) -> FrozenSet[RelationType]:
    """Return CONTRA(τ): the types that structurally contradict ``tau``.

    Types absent from the table contradict nothing structurally (empty set).
    A ``CONTRADICTS`` relator names a contradiction explicitly; CONTRA detects
    the implicit ones (MATH_SPEC §1.3).
    """
    return _CONTRA.get(tau, frozenset())


# ---------------------------------------------------------------------------
# SIMILAR clusters: types that are semantically close (used by D_type).
# CONTRA always takes precedence over SIMILAR in distance computation.
# ---------------------------------------------------------------------------

_SIMILAR: Dict[RelationType, FrozenSet[RelationType]] = {
    RelationType.CAUSAL: frozenset(
        {
            RelationType.ENABLES,
            RelationType.EVIDENCES,
            RelationType.CONDITIONAL,
            RelationType.TEMPORAL_BEFORE,
        }
    ),
    RelationType.ENABLES: frozenset({RelationType.CAUSAL, RelationType.CONDITIONAL}),
    RelationType.EVIDENCES: frozenset({RelationType.CAUSAL, RelationType.ANALOGOUS}),
    RelationType.CONDITIONAL: frozenset({RelationType.CAUSAL, RelationType.ENABLES}),
    RelationType.CONTAINS: frozenset({RelationType.INSTANTIATES, RelationType.REFINES}),
    RelationType.INSTANTIATES: frozenset({RelationType.CONTAINS, RelationType.REFINES}),
    RelationType.REFINES: frozenset(
        {
            RelationType.CONTAINS,
            RelationType.INSTANTIATES,
            RelationType.ANALOGOUS,
        }
    ),
    RelationType.NEGATES: frozenset({RelationType.CONTRADICTS, RelationType.VIOLATES}),
    RelationType.CONTRADICTS: frozenset({RelationType.NEGATES, RelationType.VIOLATES}),
    RelationType.VIOLATES: frozenset({RelationType.NEGATES, RelationType.CONTRADICTS}),
    RelationType.ANALOGOUS: frozenset({RelationType.REFINES, RelationType.EVIDENCES}),
    RelationType.TEMPORAL_BEFORE: frozenset({RelationType.CAUSAL}),
}


def similar(tau: RelationType) -> FrozenSet[RelationType]:
    """Return SIMILAR(τ): semantically-close types (excludes ``tau`` itself)."""
    return _SIMILAR.get(tau, frozenset())


def d_type(tau_a: RelationType, tau_b: RelationType) -> float:
    """Semantic distance D_type over relation types (MATH_SPEC §3.2).

    ::

        0.0  if τ_a == τ_b
        1.0  if τ_b ∈ CONTRA(τ_a)        (contradiction dominates)
        0.3  if τ_b ∈ SIMILAR(τ_a)
        0.7  otherwise (unrelated)

    CONTRA is checked before SIMILAR so a contradicting pair always scores 1.0.
    """
    if tau_a == tau_b:
        return D_TYPE_SAME
    if tau_b in contra(tau_a):
        return D_TYPE_CONTRA
    if tau_b in similar(tau_a):
        return D_TYPE_SIMILAR
    return D_TYPE_UNRELATED


# ---------------------------------------------------------------------------
# Ω composition-compatibility table (MATH_SPEC §1.2).
# Maps an ordered pair (τ₁, τ₂) → composed type + strength rule.
# Strength rules take (σ₁, σ₂) and return the composed strength.
# A pair absent from the table is non-composable.
# ---------------------------------------------------------------------------

StrengthRule = Callable[[float, float], float]


class CompositionRule:
    """A single entry of Ω: the composed type and how strength combines."""

    __slots__ = ("result_type", "strength_rule", "description")

    def __init__(
        self,
        result_type: RelationType,
        strength_rule: StrengthRule,
        description: str,
    ) -> None:
        self.result_type = result_type
        self.strength_rule = strength_rule
        self.description = description

    def __repr__(self) -> str:
        return (
            f"CompositionRule(result_type={self.result_type!r}, "
            f"rule={self.description!r})"
        )


_OMEGA: Dict[Tuple[RelationType, RelationType], CompositionRule] = {
    (RelationType.CAUSAL, RelationType.CAUSAL): CompositionRule(
        RelationType.CAUSAL, lambda s1, s2: s1 * s2, "σ₁·σ₂"
    ),
    (RelationType.CAUSAL, RelationType.ENABLES): CompositionRule(
        RelationType.ENABLES, lambda s1, s2: s1 * s2 * 0.8, "σ₁·σ₂·0.8"
    ),
    (RelationType.CONDITIONAL, RelationType.CAUSAL): CompositionRule(
        RelationType.CONDITIONAL, lambda s1, s2: min(s1, s2), "min(σ₁,σ₂)"
    ),
    (RelationType.EVIDENCES, RelationType.EVIDENCES): CompositionRule(
        RelationType.EVIDENCES,
        lambda s1, s2: 1.0 - (1.0 - s1) * (1.0 - s2),
        "1-(1-σ₁)(1-σ₂)",
    ),
    (RelationType.ANALOGOUS, RelationType.ANALOGOUS): CompositionRule(
        RelationType.REFINES, lambda s1, s2: (s1 + s2) / 2.0, "(σ₁+σ₂)/2"
    ),
}


def composition_rule(
    tau_1: RelationType, tau_2: RelationType
) -> Optional[CompositionRule]:
    """Return the Ω rule for composing ``tau_1`` then ``tau_2``, or None.

    A ``CONTRADICTS`` first operand blocks composition entirely (MATH_SPEC §1.2:
    ``CONTRADICTS | * | ∅``). Any pair not present in Ω is non-composable.
    """
    if tau_1 == RelationType.CONTRADICTS:
        return None
    return _OMEGA.get((tau_1, tau_2))


def can_compose_types(tau_1: RelationType, tau_2: RelationType) -> bool:
    """Return True iff types ``tau_1`` then ``tau_2`` are composable under Ω."""
    return composition_rule(tau_1, tau_2) is not None
