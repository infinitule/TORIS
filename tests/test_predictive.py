"""Tests for the Predictive Engine cycle (MATH_SPEC §3.1, §3.3)."""

import pytest

from toris.engine.predictive import PredictiveEngine
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {name: ConceptState(id=name) for name in ["A", "B", "C", "D"]}


def _r(concepts, s, t, tau=RelationType.CAUSAL, sigma=0.8):
    return Relator(tau, concepts[s], concepts[t], sigma=sigma)


def test_project_continuity_is_a_copy(concepts):
    f = RelationalField()
    f.add_relator(_r(concepts, "A", "B"))
    eng = PredictiveEngine()
    pred = eng.project(f)
    assert pred.edge_set() == f.edge_set()
    assert pred.relators()[0] is not f.relators()[0]  # distinct objects


def test_project_with_relevance_warps(concepts):
    f = RelationalField()
    f.add_relator(_r(concepts, "A", "B"))
    f.add_relator(_r(concepts, "C", "D"))
    eng = PredictiveEngine()
    pred = eng.project(f, relevance=lambda r: 1.0 if r.src_id == "A" else 0.0)
    assert pred.edge_set() == {("A", "B")}  # C→D suppressed by the goal


def test_observe_from_iterable(concepts):
    eng = PredictiveEngine()
    obs = eng.observe([_r(concepts, "A", "B"), _r(concepts, "B", "C")])
    assert isinstance(obs, RelationalField)
    assert obs.edge_set() == {("A", "B"), ("B", "C")}


def test_observe_passthrough_field(concepts):
    f = RelationalField()
    f.add_relator(_r(concepts, "A", "B"))
    eng = PredictiveEngine()
    assert eng.observe(f) is f


def test_propagate_records_epsilon_and_gates(concepts):
    pred = RelationalField()
    pred.add_relator(_r(concepts, "A", "B"))
    obs = RelationalField()
    confirmed = _r(concepts, "A", "B")
    novel = _r(concepts, "C", "D")
    obs.add_relator(confirmed)
    obs.add_relator(novel)

    eng = PredictiveEngine()
    report = eng.compute_delta(pred, obs)
    propagating = eng.propagate(report)

    # ε is recorded onto every observed relator
    assert confirmed.epsilon == pytest.approx(0.0)
    assert novel.epsilon == pytest.approx(report.per_relator[novel.rid].epsilon)
    # only the novel relator propagates
    assert [r.rid for r in propagating] == [novel.rid]


def test_step_runs_full_cycle(concepts):
    field = RelationalField()
    field.add_relator(_r(concepts, "A", "B"))
    eng = PredictiveEngine()
    # observe the same structure plus one novelty
    report, propagating = eng.step(
        field, [_r(concepts, "A", "B"), _r(concepts, "C", "D")]
    )
    assert report.num_processing_events() == 1
    assert {r.edge for r in propagating} == {("C", "D")}
