"""Tests for Layer 7 — TASF contour integration and pole detection."""

import pytest
from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.complex_salience import ComplexSalienceField, encode_complex
from toris.engine.tasf import TASF, TASFReport


def _concept(name):
    return ConceptState(id=name)


def _relator(src, tgt, tau, sigma=0.8, kappa=0.7, epsilon=0.1):
    return Relator(tau=tau, src=src, tgt=tgt, sigma=sigma, kappa=kappa, epsilon=epsilon)


def _smooth_field():
    cs = {n: _concept(n) for n in "ABCD"}
    taus = [RelationType.CAUSAL, RelationType.ENABLES, RelationType.EVIDENCES, RelationType.CONTAINS]
    f_obs = RelationalField()
    f_pred = RelationalField()
    for c in cs.values():
        f_obs.add_concept(c)
        f_pred.add_concept(c)
    keys = list("ABCD")
    for i, (tau) in enumerate(taus):
        src, tgt = cs[keys[i % 4]], cs[keys[(i + 1) % 4]]
        r = _relator(src, tgt, tau, kappa=0.6 + 0.05 * i, epsilon=0.02)
        f_obs.add_relator(r)
        f_pred.add_relator(r.clone(epsilon=0.0))
    return f_obs, f_pred


def _contradiction_field():
    cs = {n: _concept(n) for n in "XYZ"}
    f_obs = RelationalField()
    f_pred = RelationalField()
    for c in cs.values():
        f_obs.add_concept(c)
        f_pred.add_concept(c)
    r_pred = _relator(cs["X"], cs["Y"], RelationType.CAUSAL, sigma=0.8, kappa=0.5, epsilon=0.0)
    r_obs = _relator(cs["X"], cs["Y"], RelationType.NEGATES, sigma=0.7, kappa=0.5, epsilon=0.8)
    r_norm = _relator(cs["Y"], cs["Z"], RelationType.ENABLES, sigma=0.75, kappa=0.65, epsilon=0.05)
    f_pred.add_relator(r_pred.clone())
    f_pred.add_relator(r_norm.clone(epsilon=0.0))
    f_obs.add_relator(r_obs)
    f_obs.add_relator(r_norm)
    return f_obs, f_pred


class TestComplexSalienceField:
    def test_encode_complex_on_real_axis(self):
        k = encode_complex(0.5)
        assert k.real == pytest.approx(0.5)
        assert k.imag == pytest.approx(0.0)

    def test_F_directed_smooth_is_small(self):
        f_obs, f_pred = _smooth_field()
        csf = ComplexSalienceField()
        val = csf.F_directed(f_obs, f_pred, complex(0.5, 0))
        # Smooth field has tiny type mismatch
        assert abs(val) < 0.5

    def test_F_directed_has_pole_for_contradiction(self):
        f_obs, f_pred = _contradiction_field()
        csf = ComplexSalienceField()
        # Evaluate far from pole (κ=0.9) and near pole (κ=0.5+tiny imag)
        val_far = csf.F_directed(f_obs, f_pred, complex(0.9, 0.0))
        val_near = csf.F_directed(f_obs, f_pred, complex(0.5, 1e-5))
        # Near the pole the imaginary part should be large
        assert abs(val_near.imag) > abs(val_far.imag)

    def test_W_goal_unity_without_goal(self):
        csf = ComplexSalienceField()
        w = csf.W_goal(complex(0.5, 0), None)
        assert w == pytest.approx(1.0 + 0j)


class TestTASF:
    def test_quadrature_points_on_unit_circle(self):
        tasf = TASF(N_quadrature=8, kappa_max=1.0)
        pts = tasf.gaussian_quadrature_circle()
        assert len(pts) == 8
        for pt in pts:
            assert abs(abs(pt) - 1.0) < 1e-10

    def test_double_zero_weight_at_kappa_max(self):
        tasf = TASF()
        w = tasf.double_zero_weight(complex(1.0, 0))
        assert abs(w) < 1e-10

    def test_smooth_field_delta_S_small(self):
        f_obs, f_pred = _smooth_field()
        tasf = TASF(N_quadrature=32)
        report = tasf.compute(f_obs, f_pred)
        assert isinstance(report, TASFReport)
        # Smooth field: analytic ΔS should be finite
        assert abs(report.delta_S_analytic) < 10.0

    def test_contradiction_field_poles_detected(self):
        f_obs, f_pred = _contradiction_field()
        tasf = TASF(N_quadrature=32)
        report = tasf.compute(f_obs, f_pred)
        assert len(report.poles) >= 1

    def test_residues_nonzero_for_contradiction(self):
        f_obs, f_pred = _contradiction_field()
        tasf = TASF(N_quadrature=32)
        report = tasf.compute(f_obs, f_pred)
        if report.poles:
            assert any(abs(r) > 0 for r in report.residues)

    def test_report_fields_present(self):
        f_obs, f_pred = _smooth_field()
        tasf = TASF()
        report = tasf.compute(f_obs, f_pred)
        assert hasattr(report, "delta_S_analytic")
        assert hasattr(report, "poles")
        assert hasattr(report, "residues")
        assert hasattr(report, "N_quadrature")
        assert report.N_quadrature == 32
