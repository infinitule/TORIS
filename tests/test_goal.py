"""Tests for the Goal Manifold, relevance function, and warp Φ (MATH_SPEC §4)."""

import pytest

from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.goal.subgoal import Subgoal, SubgoalStatus
from toris.goal.warp import (
    concept_overlap,
    goal_relevance_multiplier,
    relevance,
    type_fit,
)
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {n: ConceptState(id=n) for n in ["A", "B", "C", "D"]}


def _r(concepts, s, t, tau=RelationType.CAUSAL, kappa=1.0):
    return Relator(tau, concepts[s], concepts[t], kappa=kappa)


# -- relevance function (MATH_SPEC §4.3) ------------------------------------


def test_concept_overlap(concepts):
    r = _r(concepts, "A", "B")
    assert concept_overlap(r, Goal("g", concepts=set())) == 1.0  # unconstrained
    assert concept_overlap(r, Goal("g", concepts={"A", "B"})) == 1.0
    assert concept_overlap(r, Goal("g", concepts={"A"})) == 0.5
    assert concept_overlap(r, Goal("g", concepts={"C", "D"})) == 0.0


def test_type_fit(concepts):
    g_empty = Goal("g", preferred_types=set())
    assert type_fit(RelationType.CAUSAL, g_empty) == 1.0  # neutral
    g_causal = Goal("g", preferred_types={RelationType.CAUSAL})
    assert type_fit(RelationType.CAUSAL, g_causal) == 1.0  # preferred
    # ENABLES is SIMILAR to CAUSAL (D_type=0.3) → fit 0.7
    assert type_fit(RelationType.ENABLES, g_causal) == pytest.approx(0.7)
    # NEGATES ∈ CONTRA(CAUSAL) (D_type=1.0) → fit 0.0
    assert type_fit(RelationType.NEGATES, g_causal) == pytest.approx(0.0)
    # VIOLATES unrelated to CAUSAL (D_type=0.7) → fit 0.3
    assert type_fit(RelationType.VIOLATES, g_causal) == pytest.approx(0.3)


def test_relevance_is_product(concepts):
    r = _r(concepts, "A", "B", RelationType.ENABLES)
    g = Goal("g", concepts={"A"}, preferred_types={RelationType.CAUSAL})
    # overlap 0.5 × type_fit 0.7 = 0.35
    assert relevance(r, g) == pytest.approx(0.35)


# -- salience multiplier (MATH_SPEC §4.2 step 1) ----------------------------


def test_multiplier_without_subgoals_uses_primary_alone(concepts):
    r = _r(concepts, "A", "B")
    m = GoalManifold(
        Goal("g", concepts={"A", "B"}, preferred_types={RelationType.CAUSAL})
    )
    # no active subgoals → subgoal factor defaults to 1.0
    assert goal_relevance_multiplier(m, r) == pytest.approx(1.0)


def test_multiplier_with_subgoal(concepts):
    r = _r(concepts, "A", "B")
    primary = Goal("g", concepts={"A", "B"}, preferred_types={RelationType.CAUSAL})
    m = GoalManifold(primary)
    m.add_subgoal(
        Subgoal(
            "s",
            priority=0.5,
            concepts={"A", "B"},
            preferred_types={RelationType.CAUSAL},
        )
    )
    # rel_primary=1.0 ; subgoal factor = 0.5 × rel_sub(1.0) = 0.5 → m = 0.5
    assert goal_relevance_multiplier(m, r) == pytest.approx(0.5)


# -- subgoal stack management ----------------------------------------------


def test_subgoal_lifecycle():
    m = GoalManifold(Goal("g"))
    s = m.add_subgoal(Subgoal("explore", priority=0.8))
    assert s in m.active and s.status is SubgoalStatus.ACTIVE
    m.resolve_subgoal(s)
    assert s not in m.active and s in m.resolved
    assert s.status is SubgoalStatus.RESOLVED

    s2 = m.add_subgoal(Subgoal("dead-end"))
    m.abandon_subgoal(s2)
    assert s2 in m.abandoned and s2.status is SubgoalStatus.ABANDONED


def test_priority_clamped():
    s = Subgoal("x", priority=5.0)
    assert s.priority == 1.0


# -- the warp operator Φ ----------------------------------------------------


def test_warp_suppresses_out_of_scope_and_amplifies_in_scope(concepts):
    f = RelationalField()
    in_scope = _r(concepts, "A", "B", RelationType.CAUSAL)  # in goal scope
    out_scope = _r(concepts, "C", "D", RelationType.CAUSAL)  # out of scope
    f.add_relator(in_scope)
    f.add_relator(out_scope)

    m = GoalManifold(
        Goal("g", concepts={"A", "B"}, preferred_types={RelationType.CAUSAL})
    )
    warped = m.warp(f)
    assert ("A", "B") in warped.edge_set()
    assert ("C", "D") not in warped.edge_set()  # suppressed (κ' = 0)


def test_warp_surfaces_contradiction_to_log(concepts):
    f = RelationalField()
    # in-scope contradiction on A→B; both types fit the goal so both survive
    f.add_relator(_r(concepts, "A", "B", RelationType.CAUSAL))
    f.add_relator(_r(concepts, "A", "B", RelationType.NEGATES))
    m = GoalManifold(
        Goal(
            "g",
            concepts={"A", "B"},
            preferred_types={RelationType.CAUSAL, RelationType.NEGATES},
        )
    )
    m.warp(f)
    assert len(m.contradiction_log) == 1
    assert len(m.contradiction_log.live()) == 1


def test_warp_does_not_surface_suppressed_contradiction(concepts):
    f = RelationalField()
    # contradiction out of the goal's scope → both suppressed → not surfaced
    f.add_relator(_r(concepts, "C", "D", RelationType.CAUSAL))
    f.add_relator(_r(concepts, "C", "D", RelationType.NEGATES))
    m = GoalManifold(Goal("g", concepts={"A", "B"}))
    m.warp(f)
    assert len(m.contradiction_log) == 0
