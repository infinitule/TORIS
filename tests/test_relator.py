"""Tests for the Relator primitive and its algebra operators (§1.1–1.3)."""

import pytest

from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {
        "A": ConceptState(id="A"),
        "B": ConceptState(id="B"),
        "C": ConceptState(id="C"),
    }


def test_construction_and_fields(concepts):
    r = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"], sigma=0.7)
    assert r.tau == RelationType.CAUSAL
    assert r.edge == ("A", "B")
    assert r.src_id == "A" and r.tgt_id == "B"
    assert r.sigma == pytest.approx(0.7)
    assert r.kappa == pytest.approx(1.0)
    assert r.epsilon == pytest.approx(0.0)


def test_parameters_clamped(concepts):
    r = Relator(
        RelationType.CAUSAL,
        concepts["A"],
        concepts["B"],
        sigma=5.0,
        kappa=-1.0,
        epsilon=-3.0,
    )
    assert r.sigma == 1.0
    assert r.kappa == 0.0
    assert r.epsilon == 0.0


def test_unique_identities(concepts):
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    assert r1.rid != r2.rid


# -- symmetry / reverse -----------------------------------------------------


def test_symmetric_relator_reverses(concepts):
    r = Relator(RelationType.ANALOGOUS, concepts["A"], concepts["B"], sigma=0.8)
    assert r.is_symmetric
    inv = r.reverse()
    assert inv.edge == ("B", "A")
    assert inv.tau == RelationType.ANALOGOUS
    assert inv.sigma == pytest.approx(0.8)
    assert inv.rid != r.rid


def test_asymmetric_relator_cannot_reverse(concepts):
    r = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    assert not r.is_symmetric
    with pytest.raises(ValueError):
        r.reverse()


# -- contradiction operator ⊗ ----------------------------------------------


def test_contradicts_same_pair_via_contra(concepts):
    # CAUSAL(A→B) vs NEGATES(A→B): NEGATES ∈ CONTRA(CAUSAL) → contradiction.
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.NEGATES, concepts["A"], concepts["B"])
    assert r1.contradicts(r2)
    assert r2.contradicts(r1)  # detection is order-independent


def test_no_contradiction_on_different_pairs(concepts):
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.NEGATES, concepts["A"], concepts["C"])
    assert not r1.contradicts(r2)


def test_explicit_contradicts_relator(concepts):
    r1 = Relator(RelationType.CONTRADICTS, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    assert r1.contradicts(r2)
    # two CONTRADICTS on the same pair do not contradict each other
    r3 = Relator(RelationType.CONTRADICTS, concepts["A"], concepts["B"])
    assert not r1.contradicts(r3)


def test_compatible_relators_do_not_contradict(concepts):
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.ENABLES, concepts["A"], concepts["B"])
    assert not r1.contradicts(r2)


# -- composition operator ∘_τ ----------------------------------------------


def test_compose_causal_chain(concepts):
    # A→B (CAUSAL, .9) ∘ B→C (CAUSAL, .8) = A→C (CAUSAL, .72)
    r1 = Relator(
        RelationType.CAUSAL, concepts["A"], concepts["B"], sigma=0.9, epsilon=0.1
    )
    r2 = Relator(
        RelationType.CAUSAL, concepts["B"], concepts["C"], sigma=0.8, epsilon=0.2
    )
    comp = r1.compose(r2)
    assert comp is not None
    assert comp.tau == RelationType.CAUSAL
    assert comp.edge == ("A", "C")
    assert comp.sigma == pytest.approx(0.72)
    # surprise accumulates additively along the chain
    assert comp.epsilon == pytest.approx(0.3)


def test_compose_kappa_is_min(concepts):
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"], kappa=0.4)
    r2 = Relator(RelationType.CAUSAL, concepts["B"], concepts["C"], kappa=0.9)
    comp = r1.compose(r2)
    assert comp.kappa == pytest.approx(0.4)


def test_compose_requires_shared_concept(concepts):
    # tgt(r1)=B but src(r2)=A → undefined.
    r1 = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.CAUSAL, concepts["A"], concepts["C"])
    assert not r1.can_compose(r2)
    assert r1.compose(r2) is None


def test_contradicts_first_operand_blocks_compose(concepts):
    r1 = Relator(RelationType.CONTRADICTS, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.CAUSAL, concepts["B"], concepts["C"])
    assert not r1.can_compose(r2)
    assert r1.compose(r2) is None


def test_uncomposable_types_return_none(concepts):
    r1 = Relator(RelationType.CONTAINS, concepts["A"], concepts["B"])
    r2 = Relator(RelationType.NEGATES, concepts["B"], concepts["C"])
    assert r1.compose(r2) is None


# -- history ----------------------------------------------------------------


def test_register_records_in_both_endpoints(concepts):
    r = Relator(RelationType.CAUSAL, concepts["A"], concepts["B"])
    r.register()
    assert r in concepts["A"].relational_history
    assert r in concepts["B"].relational_history


def test_register_self_loop_records_once():
    a = ConceptState(id="A")
    r = Relator(RelationType.REFINES, a, a)
    r.register()
    assert concepts_count(a, r) == 1


def concepts_count(concept, relator):
    return concept.relational_history.count(relator)
