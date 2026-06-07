"""Tests for the ContradictionLog and PRODUCTIVE status (CLAUDE.md §3.5)."""

import pytest

from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.contradiction import ContradictionLog, ResolutionStatus


@pytest.fixture
def concepts():
    return {name: ConceptState(id=name) for name in ["A", "B", "C"]}


def _r(concepts, s, t, tau):
    return Relator(tau, concepts[s], concepts[t])


def test_detect_contradiction(concepts):
    a = _r(concepts, "A", "B", RelationType.CAUSAL)
    b = _r(concepts, "A", "B", RelationType.NEGATES)
    assert ContradictionLog.detect_contradiction(a, b)
    c = _r(concepts, "A", "B", RelationType.ENABLES)
    assert not ContradictionLog.detect_contradiction(a, c)


def test_log_contradiction_and_dedupe(concepts):
    log = ContradictionLog()
    a = _r(concepts, "A", "B", RelationType.CAUSAL)
    b = _r(concepts, "A", "B", RelationType.NEGATES)
    e1 = log.log_contradiction(a, b, t_discovered=3)
    e2 = log.log_contradiction(a, b, t_discovered=99)  # same pair
    assert e1 is e2  # idempotent on the pair
    assert len(log) == 1
    assert e1.t_discovered == 3
    assert e1.resolution_status is ResolutionStatus.LIVE


def test_log_non_contradiction_raises(concepts):
    log = ContradictionLog()
    a = _r(concepts, "A", "B", RelationType.CAUSAL)
    b = _r(concepts, "A", "B", RelationType.ENABLES)
    with pytest.raises(ValueError):
        log.log_contradiction(a, b)


def test_mark_productive_is_held_and_sticky(concepts):
    log = ContradictionLog()
    a = _r(concepts, "A", "B", RelationType.EVIDENCES)
    b = _r(concepts, "A", "B", RelationType.NEGATES)
    log.log_contradiction(a, b)
    entry = log.mark_productive(a, b, note="wave/particle")
    assert entry.is_productive
    assert entry.is_held
    assert entry.note == "wave/particle"
    # re-logging the same pair must NOT downgrade PRODUCTIVE back to LIVE
    again = log.log_contradiction(a, b)
    assert again.resolution_status is ResolutionStatus.PRODUCTIVE


def test_status_queries(concepts):
    log = ContradictionLog()
    a = _r(concepts, "A", "B", RelationType.CAUSAL)
    b = _r(concepts, "A", "B", RelationType.NEGATES)
    c = _r(concepts, "B", "C", RelationType.CONTAINS)
    d = _r(concepts, "B", "C", RelationType.VIOLATES)
    log.log_contradiction(a, b)
    log.log_contradiction(c, d)
    log.mark_productive(a, b)
    assert len(log.productive()) == 1
    assert len(log.live()) == 1
    assert len(log.held()) == 2  # productive + live both held
    log.mark_resolved(c, d)
    assert len(log.held()) == 1  # resolved no longer held


def test_scan_field_surfaces_parallel_contradictions(concepts):
    f = RelationalField()
    f.add_relator(_r(concepts, "A", "B", RelationType.CAUSAL))
    f.add_relator(_r(concepts, "A", "B", RelationType.NEGATES))  # contradicts
    f.add_relator(_r(concepts, "B", "C", RelationType.CAUSAL))  # lone, no parallel
    log = ContradictionLog()
    touched = log.scan_field(f, t_discovered=7)
    assert len(touched) == 1
    assert len(log) == 1
    assert touched[0].t_discovered == 7
