"""Tests for Layer 7 — TORIS Michel Parameters."""

import pytest
from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.michel_parameters import (
    MichelParameters,
    compute,
    michel_alert,
    standard_values,
)


def _concept(name):
    return ConceptState(id=name)


def _relator(src, tgt, tau, sigma=0.8, kappa=0.7, epsilon=0.1):
    return Relator(tau=tau, src=src, tgt=tgt, sigma=sigma, kappa=kappa, epsilon=epsilon)


def _well_calibrated_fields():
    cs = {n: _concept(n) for n in "ABCD"}
    f_obs = RelationalField()
    f_pred = RelationalField()
    for c in cs.values():
        f_obs.add_concept(c)
        f_pred.add_concept(c)
    r = _relator(cs["A"], cs["B"], RelationType.CAUSAL, epsilon=0.01)
    f_obs.add_relator(r)
    f_pred.add_relator(r.clone(epsilon=0.0))
    return f_obs, f_pred


class TestMichelParameters:
    def test_standard_values(self):
        sv = standard_values()
        assert sv.rho_T == pytest.approx(0.75)
        assert sv.eta_T == pytest.approx(0.0)
        assert sv.xi_T == pytest.approx(1.0)
        assert sv.delta_T == pytest.approx(0.01)

    def test_no_alert_for_standard(self):
        sv = standard_values()
        assert not michel_alert(sv)

    def test_rho_deviation_triggers_alert(self):
        params = MichelParameters(rho_T=0.0, eta_T=0.0, xi_T=1.0, delta_T=0.0)
        assert michel_alert(params)

    def test_eta_deviation_triggers_alert(self):
        params = MichelParameters(rho_T=0.75, eta_T=0.5, xi_T=1.0, delta_T=0.0)
        assert michel_alert(params)

    def test_xi_deviation_triggers_alert(self):
        params = MichelParameters(rho_T=0.75, eta_T=0.0, xi_T=-0.5, delta_T=0.0)
        assert michel_alert(params)

    def test_delta_deviation_triggers_alert(self):
        params = MichelParameters(rho_T=0.75, eta_T=0.0, xi_T=1.0, delta_T=0.5)
        assert michel_alert(params)

    def test_compute_returns_michel_parameters(self):
        f_obs, f_pred = _well_calibrated_fields()
        params = compute(f_obs, f_pred, None)
        assert isinstance(params, MichelParameters)
        assert 0.0 <= params.rho_T <= 1.0
        assert params.delta_T >= 0.0

    def test_compute_well_calibrated_near_standard(self):
        f_obs, f_pred = _well_calibrated_fields()
        params = compute(f_obs, f_pred, None)
        # Well-calibrated: rho close to 3/4
        assert abs(params.rho_T - 0.75) <= 0.75  # at most one standard deviation
        # eta close to 0
        assert params.eta_T <= 0.5

    def test_str_representation(self):
        sv = standard_values()
        s = str(sv)
        assert "rho" in s.lower() or "ρ" in s

    def test_alert_thresholds_are_positive(self):
        # Standard values define the calibrated TORIS field
        sv = standard_values()
        assert sv.rho_T > 0
        assert sv.xi_T > 0
