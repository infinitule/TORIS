"""The goal-driven field warp operator Φ (MATH_SPEC §4.2–4.3).

This module supplies the *policy* of the warp — the relevance function (§4.3)
and the salience-recomputation multiplier (§4.2 step 1) — and drives the field's
warp *mechanism* (``RelationalField.warp``, which performs steps 1–3). It then
performs step 4: surfacing the contradictions in the warped (active) topology to
the manifold's contradiction log.

    Φ(G, F) → F'

The relevance function is the one component the spec marks as "learned" in the
full system; here it is the reference initialization:

    relevance(R, g) = concept_overlap(R, g) × type_fit(τ(R), g)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Protocol, Set

from toris.constants import THETA_AMPLIFY, THETA_KAPPA
from toris.field.relational_field import RelationalField
from toris.primitives.relation_types import RelationType, d_type
from toris.primitives.relator import Relator

if TYPE_CHECKING:  # avoid a manifold<->warp import cycle at runtime
    from toris.goal.manifold import GoalManifold


class GoalLike(Protocol):
    """Anything the relevance function can score: a Goal or a Subgoal."""

    concepts: Set[str]
    preferred_types: Set[RelationType]


# ---------------------------------------------------------------------------
# The relevance function (MATH_SPEC §4.3)
# ---------------------------------------------------------------------------


def concept_overlap(relator: Relator, goal: GoalLike) -> float:
    """Does src(R) or tgt(R) appear in the goal's concept set? ∈ [0,1].

    Graded: the fraction of the relator's two endpoints that lie in the goal's
    scope (0, 0.5, or 1.0). An empty concept set means the goal does not
    constrain by concept, so overlap defaults to 1.0 (neutral).
    """
    if not goal.concepts:
        return 1.0
    endpoints = {relator.src_id, relator.tgt_id}
    return len(endpoints & goal.concepts) / 2.0


def type_fit(tau: RelationType, goal: GoalLike) -> float:
    """Is τ(R) a type useful for achieving the goal? ∈ [0,1].

    1.0 if the goal names no preferred types (neutral) or τ is preferred;
    otherwise the best semantic fit ``max_p (1 − D_type(p, τ))`` over the
    preferred types — a type close to a preferred one fits better than an
    unrelated or contradicting one (reuses the type algebra, MATH_SPEC §3.2).

    D_type is evaluated *from* the preferred type p (the type the goal wants)
    so that a relator whose type contradicts the goal's desired type is scored
    as a poor fit — D_type is directional because CONTRA is (e.g. NEGATES ∈
    CONTRA(CAUSAL) but not vice-versa).
    """
    if not goal.preferred_types or tau in goal.preferred_types:
        return 1.0
    return max(1.0 - d_type(p, tau) for p in goal.preferred_types)


def relevance(relator: Relator, goal: GoalLike) -> float:
    """relevance(R, g) = concept_overlap(R, g) × type_fit(τ(R), g) (§4.3)."""
    return concept_overlap(relator, goal) * type_fit(relator.tau, goal)


# ---------------------------------------------------------------------------
# Salience recomputation multiplier (MATH_SPEC §4.2 step 1)
# ---------------------------------------------------------------------------


def goal_relevance_multiplier(manifold: "GoalManifold", relator: Relator) -> float:
    """The factor multiplying κ(R) in step 1 of Φ.

    ``relevance(R, G_p) · Σ_{g∈S_active} [priority(g) · relevance(R, g)]``.

    When no subgoals are active the subgoal sum would be 0 and collapse the
    whole field; instead it defaults to 1.0 so the primary goal alone drives the
    warp (documented deviation D-09).
    """
    m_primary = relevance(relator, manifold.primary)
    active = manifold.active
    if active:
        subgoal_factor = sum(g.priority * relevance(relator, g) for g in active)
    else:
        subgoal_factor = 1.0
    return m_primary * subgoal_factor


def relevance_fn(manifold: "GoalManifold") -> Callable[[Relator], float]:
    """Bind a manifold into a relator→multiplier callable for ``field.warp``."""
    return lambda r: goal_relevance_multiplier(manifold, r)


# ---------------------------------------------------------------------------
# The full warp operator Φ (steps 1–4)
# ---------------------------------------------------------------------------


def warp_field(
    manifold: "GoalManifold",
    field: RelationalField,
    theta_kappa: float = THETA_KAPPA,
    theta_amplify: float = THETA_AMPLIFY,
    t_discovered: Optional[int] = None,
) -> RelationalField:
    """Φ(G, F) → F': warp ``field`` under ``manifold`` and surface contradictions.

    Steps 1–3 (recompute κ', suppress, amplify) are performed by the field's own
    ``warp`` using the goal-relevance multiplier. Step 4 scans the resulting
    active topology and logs every surviving contradiction to the manifold's
    contradiction log. Returns the warped field F'; the input field is untouched.
    """
    warped = field.warp(relevance_fn(manifold), theta_kappa, theta_amplify)
    t = manifold.t if t_discovered is None else t_discovered
    manifold.contradiction_log.scan_field(warped, t_discovered=t)
    return warped
