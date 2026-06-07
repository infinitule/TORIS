"""Tests for ConceptState: the role distribution Π and context update (§2)."""

import pytest

from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType


def _approx_sum_one(dist):
    return sum(dist.values()) == pytest.approx(1.0)


def test_default_distribution_is_uniform_simplex():
    c = ConceptState(id="cat")
    assert _approx_sum_one(c.pi)
    vals = set(round(v, 12) for v in c.pi.values())
    assert len(vals) == 1  # all equal → uniform
    assert set(c.pi.keys()) == set(RelationType)


def test_partial_distribution_is_normalized():
    c = ConceptState(
        id="cat",
        role_distribution={
            RelationType.CONTAINS: 0.8,
            RelationType.ENABLES: 0.4,
            RelationType.CAUSAL: 0.1,
        },
    )
    assert _approx_sum_one(c.pi)
    # ordering of mass preserved after normalization
    assert c.role_prob(RelationType.CONTAINS) > c.role_prob(RelationType.ENABLES)
    assert c.role_prob(RelationType.ENABLES) > c.role_prob(RelationType.CAUSAL)
    assert c.role_prob(RelationType.VIOLATES) == 0.0


def test_negative_weights_clamped():
    c = ConceptState(
        id="x",
        role_distribution={
            RelationType.CAUSAL: 1.0,
            RelationType.NEGATES: -5.0,
        },
    )
    assert c.role_prob(RelationType.NEGATES) == 0.0
    assert c.role_prob(RelationType.CAUSAL) == pytest.approx(1.0)


def test_bayesian_context_update_is_pure():
    c = ConceptState(
        id="cat",
        role_distribution={
            RelationType.CONTAINS: 0.5,
            RelationType.CAUSAL: 0.5,
        },
    )
    before = dict(c.pi)
    salience = {RelationType.CONTAINS: 1.0, RelationType.CAUSAL: 0.0}
    warped = c.goal_warped_roles(salience)
    # goal_warped_roles does not mutate
    assert c.pi == before
    # zeroing CAUSAL salience moves all mass to CONTAINS
    assert warped[RelationType.CONTAINS] == pytest.approx(1.0)
    assert warped[RelationType.CAUSAL] == pytest.approx(0.0)
    assert _approx_sum_one(warped)


def test_update_role_distribution_mutates():
    c = ConceptState(
        id="cat",
        role_distribution={
            RelationType.CONTAINS: 0.5,
            RelationType.CAUSAL: 0.5,
        },
    )
    salience = {RelationType.CONTAINS: 0.0, RelationType.CAUSAL: 1.0}
    result = c.update_role_distribution(salience)
    assert c.pi == result
    assert c.role_prob(RelationType.CAUSAL) == pytest.approx(1.0)


def test_bayesian_update_matches_formula():
    # Π^G(τ) = Π(τ)·ψ(τ) / Σ Π·ψ
    c = ConceptState(
        id="x",
        role_distribution={
            RelationType.CAUSAL: 0.6,
            RelationType.ENABLES: 0.4,
        },
    )
    psi = {RelationType.CAUSAL: 0.5, RelationType.ENABLES: 1.0}
    # numerators: 0.6*0.5=0.30, 0.4*1.0=0.40 ; Z=0.70
    warped = c.goal_warped_roles(psi)
    assert warped[RelationType.CAUSAL] == pytest.approx(0.30 / 0.70)
    assert warped[RelationType.ENABLES] == pytest.approx(0.40 / 0.70)


def test_all_zero_salience_falls_back_to_uniform():
    c = ConceptState(id="x", role_distribution={RelationType.CAUSAL: 1.0})
    warped = c.goal_warped_roles({t: 0.0 for t in RelationType})
    assert _approx_sum_one(warped)
    assert warped[RelationType.CAUSAL] == pytest.approx(1.0 / len(RelationType))


def test_role_distance_is_metric_in_unit_interval():
    c = ConceptState(
        id="contract",
        role_distribution={
            RelationType.CONTAINS: 0.4,
            RelationType.VIOLATES: 0.3,
            RelationType.CONDITIONAL: 0.3,
        },
    )
    g1 = {
        RelationType.CONTAINS: 1.0,
        RelationType.VIOLATES: 0.1,
        RelationType.CONDITIONAL: 0.1,
    }
    g2 = {
        RelationType.CONTAINS: 0.1,
        RelationType.VIOLATES: 1.0,
        RelationType.CONDITIONAL: 0.1,
    }
    d = c.role_distance(g1, g2)
    assert 0.0 <= d <= 1.0
    # identical contexts → zero distance
    assert c.role_distance(g1, g1) == pytest.approx(0.0, abs=1e-9)
    # different contexts → strictly positive distance
    assert d > 0.0


def test_goal_salience_clamped():
    c = ConceptState(id="x")
    c.set_salience("G1", 5.0)
    c.set_salience("G2", -1.0)
    assert c.salience_for("G1") == 1.0
    assert c.salience_for("G2") == 0.0
    assert c.salience_for("unknown") == 0.0


def test_dominant_role():
    c = ConceptState(
        id="x",
        role_distribution={
            RelationType.CAUSAL: 0.7,
            RelationType.ENABLES: 0.3,
        },
    )
    assert c.dominant_role() == RelationType.CAUSAL
