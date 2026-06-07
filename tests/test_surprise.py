"""Tests for the topological surprise metric ΔS (MATH_SPEC §3)."""

import pytest

from toris.constants import ALPHA, BETA
from toris.engine.surprise import SurpriseMetric
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {name: ConceptState(id=name) for name in ["A", "B", "C", "D"]}


def _field(relators):
    f = RelationalField()
    f.add_relators(relators)
    return f


def _r(concepts, s, t, tau=RelationType.CAUSAL, sigma=0.8):
    return Relator(tau, concepts[s], concepts[t], sigma=sigma)


def test_structural_delta_symmetric_difference(concepts):
    # pred: A→B, B→C ; obs: B→C, C→D
    pred = _field([_r(concepts, "A", "B"), _r(concepts, "B", "C")])
    obs = _field([_r(concepts, "B", "C"), _r(concepts, "C", "D")])
    m = SurpriseMetric()
    # |E_obs∖E_pred| = 1 (C→D), |E_pred∖E_obs| = 1 (A→B), |E_pred| = 2
    assert m.structural_delta(
        pred.relator_index(), obs.relator_index()
    ) == pytest.approx((1 + 1) / (2 + 1))


def test_type_delta_over_matched_edges(concepts):
    # same edge A→B, predicted CAUSAL, observed NEGATES → D_type = 1.0
    pred = _field([_r(concepts, "A", "B", RelationType.CAUSAL)])
    obs = _field([_r(concepts, "A", "B", RelationType.NEGATES)])
    m = SurpriseMetric()
    # one matched edge, D_type=1.0, denominator |E_match|+1 = 2
    assert m.type_delta(pred.relator_index(), obs.relator_index()) == pytest.approx(
        1.0 / 2
    )


def test_strength_delta_mse_over_matched(concepts):
    pred = _field([_r(concepts, "A", "B", sigma=0.9)])
    obs = _field([_r(concepts, "A", "B", sigma=0.5)])
    m = SurpriseMetric()
    # (0.9-0.5)^2 = 0.16 ; /(1+1)
    assert m.strength_delta(pred.relator_index(), obs.relator_index()) == pytest.approx(
        0.16 / 2
    )


def test_aggregate_delta_s_weighting(concepts):
    pred = _field([_r(concepts, "A", "B")])
    obs = _field([_r(concepts, "A", "B"), _r(concepts, "C", "D")])
    m = SurpriseMetric()
    pi, oi = pred.relator_index(), obs.relator_index()
    expected = (
        m.alpha * m.structural_delta(pi, oi)
        + m.beta * m.type_delta(pi, oi)
        + m.gamma * m.strength_delta(pi, oi)
    )
    assert m.topological_surprise(pi, oi) == pytest.approx(expected)


# -- per-relator surprise ε(R) (MATH_SPEC §3.3) -----------------------------


def test_unpredicted_relator_has_structural_surprise(concepts):
    pred = _field([])
    obs = _field([_r(concepts, "A", "B")])
    m = SurpriseMetric()
    rs = m.relator_surprise(obs.relators()[0], pred.relator_index())
    assert not rs.predicted
    assert rs.eps_struct == 1.0
    assert rs.epsilon == pytest.approx(ALPHA)  # α·1 + β·0 + γ·0
    assert rs.propagates()  # 0.6 > 0.2


def test_confirmed_prediction_is_suppressed(concepts):
    r_pred = _r(concepts, "A", "B", RelationType.CAUSAL, sigma=0.8)
    r_obs = _r(concepts, "A", "B", RelationType.CAUSAL, sigma=0.8)
    pred = _field([r_pred])
    obs = _field([r_obs])
    m = SurpriseMetric()
    rs = m.relator_surprise(obs.relators()[0], pred.relator_index())
    assert rs.predicted
    assert rs.epsilon == pytest.approx(0.0)
    assert not rs.propagates()  # confirmed → no compute


def test_wrong_type_surprise(concepts):
    pred = _field([_r(concepts, "A", "B", RelationType.CAUSAL)])
    obs = _field([_r(concepts, "A", "B", RelationType.NEGATES)])
    m = SurpriseMetric()
    rs = m.relator_surprise(obs.relators()[0], pred.relator_index())
    # D_type(CAUSAL, NEGATES) = 1.0 → ε ≈ β·1.0 = 0.3
    assert rs.eps_type == pytest.approx(1.0)
    assert rs.epsilon == pytest.approx(BETA)
    assert rs.propagates()


def test_strength_only_surprise_does_not_propagate(concepts):
    # γ = 0.1; max strength surprise (Δσ=1) gives ε = 0.1 < θ_ε = 0.2.
    pred = _field([_r(concepts, "A", "B", sigma=1.0)])
    obs = _field([_r(concepts, "A", "B", sigma=0.0)])
    m = SurpriseMetric()
    rs = m.relator_surprise(obs.relators()[0], pred.relator_index())
    assert rs.epsilon == pytest.approx(0.1)
    assert not rs.propagates()  # strength matters least, by design


def test_report_partitions_propagating_and_suppressed(concepts):
    pred = _field([_r(concepts, "A", "B")])  # predict only A→B
    obs = _field(
        [
            _r(concepts, "A", "B"),  # confirmed → suppressed
            _r(concepts, "C", "D"),  # unpredicted → propagates
        ]
    )
    m = SurpriseMetric()
    report = m.report(pred, obs)
    assert report.num_processing_events() == 1
    assert {r.edge for r in report.propagating()} == {("C", "D")}
    assert {r.edge for r in report.suppressed()} == {("A", "B")}
