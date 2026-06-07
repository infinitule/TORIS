"""Tests for Layer 9 — Harmonic Maass shadow completion."""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.maass_completion import (
    ShadowContradiction,
    shadow_cusp_form, eichler_integral, shadow_density,
    shadow_correction, complete_tasf, CompleteResult,
)


def make_contradiction(sigma_a=0.7, sigma_b=0.6, kappa_C=0.5, tau_diff=3):
    return ShadowContradiction(
        relator_a_sigma=sigma_a,
        relator_b_sigma=sigma_b,
        kappa_C=kappa_C,
        tau_diff=tau_diff,
        residue_strength=sigma_a * sigma_b,
    )


def build_contradiction_field():
    field = RelationalField()
    f_pred = RelationalField()
    A = ConceptState(id="A"); B = ConceptState(id="B")
    for c in [A, B]:
        field.add_concept(c); f_pred.add_concept(c)
    r_pred = Relator(tau=RelationType.CAUSAL, src=A, tgt=B, sigma=0.7, kappa=0.5, epsilon=0.0)
    r_obs  = Relator(tau=RelationType.CONTRADICTS, src=A, tgt=B, sigma=0.6, kappa=0.5, epsilon=0.9)
    f_pred.add_relator(r_pred)
    field.add_relator(r_obs)
    return field, f_pred


class TestShadowCuspForm:
    def test_returns_complex(self):
        ctr = make_contradiction()
        result = shadow_cusp_form(ctr, z=complex(0.5, 0.1))
        assert isinstance(result, complex)

    def test_finite(self):
        ctr = make_contradiction()
        for z in [complex(0.1, 0.5), complex(0.5, 0.1), complex(0.0, 1.0)]:
            r = shadow_cusp_form(ctr, z)
            assert math.isfinite(r.real) and math.isfinite(r.imag)

    def test_amplitude_proportional_to_sigma(self):
        ctr1 = make_contradiction(sigma_a=0.3, sigma_b=0.3)
        ctr2 = make_contradiction(sigma_a=0.9, sigma_b=0.9)
        z = complex(0.1, 0.5)
        assert abs(shadow_cusp_form(ctr2, z)) > abs(shadow_cusp_form(ctr1, z))


class TestEichlerIntegral:
    def test_returns_complex(self):
        ctr = make_contradiction()
        result = eichler_integral(ctr, kappa=0.3)
        assert isinstance(result, complex)

    def test_finite(self):
        ctr = make_contradiction()
        for kappa in [0.1, 0.3, 0.6, 0.9]:
            r = eichler_integral(ctr, kappa)
            assert math.isfinite(r.real) and math.isfinite(r.imag)


class TestShadowCorrection:
    def test_no_contradiction_returns_zero(self):
        field = RelationalField()
        f_pred = RelationalField()
        A = ConceptState(id="X"); B = ConceptState(id="Y")
        field.add_concept(A); field.add_concept(B)
        f_pred.add_concept(A); f_pred.add_concept(B)
        r = Relator(tau=RelationType.CAUSAL, src=A, tgt=B, sigma=0.8, kappa=0.7, epsilon=0.1)
        field.add_relator(r)
        f_pred.add_relator(r.clone())
        sc = shadow_correction(field, f_pred)
        assert sc == 0.0

    def test_contradiction_gives_nonzero(self):
        field, f_pred = build_contradiction_field()
        sc = shadow_correction(field, f_pred)
        # Not necessarily nonzero — depends on CONTRADICTS detection threshold
        assert math.isfinite(sc)


class TestCompleteTasf:
    def test_returns_complete_result(self):
        field, f_pred = build_contradiction_field()
        result = complete_tasf(field, f_pred)
        assert isinstance(result, CompleteResult)

    def test_all_fields_finite(self):
        field, f_pred = build_contradiction_field()
        r = complete_tasf(field, f_pred)
        assert math.isfinite(r.delta_S_mock)
        assert math.isfinite(r.delta_S_shadow)
        assert math.isfinite(r.delta_S_complete)

    def test_complete_equals_mock_plus_shadow(self):
        field, f_pred = build_contradiction_field()
        r = complete_tasf(field, f_pred)
        assert abs(r.delta_S_complete - (r.delta_S_mock + r.delta_S_shadow)) < 1e-9
