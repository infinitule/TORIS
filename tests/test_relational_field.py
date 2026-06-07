"""Tests for the RelationalField hypergraph (CLAUDE.md §3.3 / MATH_SPEC §4)."""

import pytest

from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {name: ConceptState(id=name) for name in ["A", "B", "C", "D"]}


def _rel(concepts, src, tgt, tau=RelationType.CAUSAL, sigma=0.8, kappa=1.0):
    return Relator(tau, concepts[src], concepts[tgt], sigma=sigma, kappa=kappa)


def test_add_relator_creates_nodes_and_edge(concepts):
    f = RelationalField()
    r = _rel(concepts, "A", "B")
    f.add_relator(r)
    assert f.num_concepts() == 2
    assert f.num_relators() == 1
    assert f.has_concept("A") and f.has_concept("B")
    assert f.get_relator(r) is r
    assert r in f


def test_parallel_contradictory_relators_coexist(concepts):
    # CAUSAL(A→B) and NEGATES(A→B) must both persist — never collapsed.
    f = RelationalField()
    r1 = _rel(concepts, "A", "B", RelationType.CAUSAL)
    r2 = _rel(concepts, "A", "B", RelationType.NEGATES)
    f.add_relator(r1)
    f.add_relator(r2)
    assert f.num_relators() == 2
    between = f.relators_between("A", "B")
    assert {x.tau for x in between} == {RelationType.CAUSAL, RelationType.NEGATES}


def test_remove_relator(concepts):
    f = RelationalField()
    r = _rel(concepts, "A", "B")
    f.add_relator(r)
    f.remove_relator(r)
    assert f.num_relators() == 0
    assert f.get_relator(r.rid) is None


def test_edge_set(concepts):
    f = RelationalField()
    f.add_relator(_rel(concepts, "A", "B"))
    f.add_relator(_rel(concepts, "B", "C"))
    assert f.edge_set() == {("A", "B"), ("B", "C")}


def test_get_neighborhood_depth(concepts):
    # chain A→B→C→D ; neighborhood of B at depth 1 = {A,B,C}
    f = RelationalField()
    f.add_relator(_rel(concepts, "A", "B"))
    f.add_relator(_rel(concepts, "B", "C"))
    f.add_relator(_rel(concepts, "C", "D"))
    nb1 = f.get_neighborhood("B", depth=1)
    ids = {c.id for c in nb1.concepts()}
    assert ids == {"A", "B", "C"}
    # relators with both endpoints in {A,B,C}: A→B and B→C
    assert nb1.num_relators() == 2
    nb2 = f.get_neighborhood("B", depth=2)
    assert {c.id for c in nb2.concepts()} == {"A", "B", "C", "D"}


def test_relator_index_keeps_strongest_on_collision(concepts):
    f = RelationalField()
    weak = _rel(concepts, "A", "B", sigma=0.2)
    strong = _rel(concepts, "A", "B", sigma=0.9)
    f.add_relator(weak)
    f.add_relator(strong)
    idx = f.relator_index()
    assert idx[("A", "B")].sigma == pytest.approx(0.9)


def test_copy_preserves_rid_and_is_independent(concepts):
    f = RelationalField()
    r = _rel(concepts, "A", "B", sigma=0.5)
    f.add_relator(r)
    dup = f.copy()
    dup_r = dup.relators()[0]
    assert dup_r.rid == r.rid  # identity preserved
    assert dup_r is not r  # but a distinct object
    dup_r.sigma = 0.1
    assert r.sigma == pytest.approx(0.5)  # original untouched


# -- warp Φ (MATH_SPEC §4.2 steps 1–3) --------------------------------------


def test_warp_suppresses_low_salience(concepts):
    f = RelationalField()
    keep = _rel(concepts, "A", "B", kappa=1.0)
    drop = _rel(concepts, "C", "D", kappa=1.0)

    f.add_relator(keep)
    f.add_relator(drop)

    # relevance 1.0 for the A→B edge, 0.0 for everything else
    def relevance(r):
        return 1.0 if r.src_id == "A" else 0.0

    warped = f.warp(relevance)
    edges = warped.edge_set()
    assert ("A", "B") in edges
    assert ("C", "D") not in edges  # suppressed: κ' = 0 ≤ θ_κ
    # original field is unchanged
    assert f.num_relators() == 2


def test_warp_amplifies_high_salience(concepts):
    f = RelationalField()
    r = _rel(concepts, "A", "B", sigma=0.5, kappa=1.0)
    f.add_relator(r)

    warped = f.warp(lambda _r: 1.0)  # κ' = 1.0 > θ_amplify → σ boosted
    wr = warped.relators()[0]
    assert wr.kappa == pytest.approx(1.0)
    assert wr.sigma > 0.5  # σ' = min(1, 0.5·(1+1.0)) = 1.0
    assert wr.sigma == pytest.approx(1.0)
    assert r.sigma == pytest.approx(0.5)  # original untouched
