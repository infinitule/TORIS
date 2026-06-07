"""Tests for the typed relational algebra over T (MATH_SPEC §1)."""

import pytest

from toris.primitives.relation_types import (
    RelationType,
    SYMMETRIC_TYPES,
    can_compose_types,
    composition_rule,
    contra,
    d_type,
    is_symmetric,
    similar,
)


def test_all_twelve_types_present():
    names = {t.name for t in RelationType}
    expected = {
        "CAUSAL",
        "CONDITIONAL",
        "CONTRADICTS",
        "CONTAINS",
        "ENABLES",
        "VIOLATES",
        "ANALOGOUS",
        "REFINES",
        "TEMPORAL_BEFORE",
        "EVIDENCES",
        "NEGATES",
        "INSTANTIATES",
    }
    assert names == expected
    assert len(RelationType) == 12


def test_symmetry_declarations():
    assert SYMMETRIC_TYPES == {RelationType.CONTRADICTS, RelationType.ANALOGOUS}
    assert is_symmetric(RelationType.CONTRADICTS)
    assert is_symmetric(RelationType.ANALOGOUS)
    assert not is_symmetric(RelationType.CAUSAL)
    assert not is_symmetric(RelationType.CONTAINS)


@pytest.mark.parametrize(
    "tau,expected",
    [
        (RelationType.CAUSAL, {RelationType.NEGATES, RelationType.CONTRADICTS}),
        (RelationType.ENABLES, {RelationType.NEGATES}),
        (RelationType.EVIDENCES, {RelationType.NEGATES}),
        (RelationType.CONTAINS, {RelationType.VIOLATES}),
        (RelationType.VIOLATES, {RelationType.CONTAINS, RelationType.ENABLES}),
        (RelationType.CONDITIONAL, {RelationType.CONTRADICTS}),
        (RelationType.ANALOGOUS, set()),  # no implicit structural contradiction
    ],
)
def test_contra_table(tau, expected):
    assert set(contra(tau)) == expected


def test_d_type_same_is_zero():
    for t in RelationType:
        assert d_type(t, t) == 0.0


def test_d_type_contra_dominates_similar():
    # NEGATES ∈ CONTRA(CAUSAL) → distance 1.0, even though both are "active".
    assert d_type(RelationType.CAUSAL, RelationType.NEGATES) == 1.0


def test_d_type_similar():
    # CAUSAL vs ENABLES — the canonical SIMILAR pair from the spec.
    assert d_type(RelationType.CAUSAL, RelationType.ENABLES) == 0.3
    assert RelationType.ENABLES in similar(RelationType.CAUSAL)


def test_d_type_unrelated():
    assert d_type(RelationType.CONTAINS, RelationType.TEMPORAL_BEFORE) == 0.7


def test_d_type_in_unit_interval():
    for a in RelationType:
        for b in RelationType:
            assert 0.0 <= d_type(a, b) <= 1.0


def test_omega_composition_rules():
    r = composition_rule(RelationType.CAUSAL, RelationType.CAUSAL)
    assert r is not None and r.result_type == RelationType.CAUSAL
    assert r.strength_rule(0.5, 0.5) == pytest.approx(0.25)

    r = composition_rule(RelationType.CAUSAL, RelationType.ENABLES)
    assert r.result_type == RelationType.ENABLES
    assert r.strength_rule(1.0, 1.0) == pytest.approx(0.8)

    r = composition_rule(RelationType.CONDITIONAL, RelationType.CAUSAL)
    assert r.result_type == RelationType.CONDITIONAL
    assert r.strength_rule(0.4, 0.9) == pytest.approx(0.4)

    r = composition_rule(RelationType.EVIDENCES, RelationType.EVIDENCES)
    assert r.result_type == RelationType.EVIDENCES
    assert r.strength_rule(0.5, 0.5) == pytest.approx(0.75)

    r = composition_rule(RelationType.ANALOGOUS, RelationType.ANALOGOUS)
    assert r.result_type == RelationType.REFINES
    assert r.strength_rule(0.4, 0.6) == pytest.approx(0.5)


def test_contradicts_first_operand_blocks_composition():
    # CONTRADICTS | * | ∅ — composition blocked for any second operand.
    for t in RelationType:
        assert composition_rule(RelationType.CONTRADICTS, t) is None
        assert not can_compose_types(RelationType.CONTRADICTS, t)


def test_uncomposable_pair_returns_none():
    assert composition_rule(RelationType.CONTAINS, RelationType.NEGATES) is None
    assert not can_compose_types(RelationType.CONTAINS, RelationType.NEGATES)
