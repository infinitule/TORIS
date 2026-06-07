"""Tests for Layer 9 — Eisenstein series and dual weighting."""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.eisenstein import (
    P_series, Q_series, R_series,
    eisenstein_weights, tau_function,
    EMPIRICAL_ALPHA, EMPIRICAL_BETA, EMPIRICAL_GAMMA,
    EISENSTEIN_ALPHA, EISENSTEIN_BETA, EISENSTEIN_GAMMA,
    D_CRIT_DEFAULT,
)


def simple_field(n=4):
    field = RelationalField()
    taus = list(RelationType)
    cs = [ConceptState(id=f"E{i}") for i in range(n + 1)]
    for c in cs: field.add_concept(c)
    for i in range(n):
        field.add_relator(Relator(
            tau=taus[i % len(taus)], src=cs[i], tgt=cs[i+1],
            sigma=0.7, kappa=0.6, epsilon=0.2
        ))
    return field


class TestEisensteinSeries:
    def test_P_at_zero_q(self):
        # q=0: P = 1
        assert abs(P_series(0.0) - 1.0) < 1e-10

    def test_Q_at_zero_q(self):
        assert abs(Q_series(0.0) - 1.0) < 1e-10

    def test_R_at_zero_q(self):
        assert abs(R_series(0.0) - 1.0) < 1e-10

    def test_P_decreasing_with_q(self):
        # For q ∈ (0,1), P < 1 (quasimodular form of weight 2)
        p0 = P_series(0.0)
        p01 = P_series(0.1)
        p03 = P_series(0.3)
        assert p0 > p01

    def test_all_finite(self):
        for q in [0.0, 0.1, 0.3, 0.5]:
            assert math.isfinite(P_series(q))
            assert math.isfinite(Q_series(q))
            assert math.isfinite(R_series(q))


class TestEisensteinWeights:
    def test_shallow_returns_empirical(self):
        for d in [1, 2, 3, 4, 5]:
            a, b, g = eisenstein_weights(d, d_crit=5)
            assert abs(a - EMPIRICAL_ALPHA) < 1e-12
            assert abs(b - EMPIRICAL_BETA) < 1e-12
            assert abs(g - EMPIRICAL_GAMMA) < 1e-12

    def test_deep_returns_eisenstein(self):
        for d in [6, 7, 10, 15]:
            a, b, g = eisenstein_weights(d, d_crit=5)
            assert abs(a - EISENSTEIN_ALPHA) < 1e-12
            assert abs(b - EISENSTEIN_BETA) < 1e-12
            assert abs(g - EISENSTEIN_GAMMA) < 1e-12

    def test_weights_sum_to_one(self):
        for d in [1, 3, 5, 6, 10]:
            a, b, g = eisenstein_weights(d)
            assert abs(a + b + g - 1.0) < 1e-12

    def test_crossover_at_d_crit(self):
        w_at = eisenstein_weights(D_CRIT_DEFAULT)
        w_above = eisenstein_weights(D_CRIT_DEFAULT + 1)
        assert w_at != w_above


class TestTauFunction:
    def test_returns_complex(self):
        field = simple_field()
        result = tau_function(field, d=3)
        assert isinstance(result, complex)

    def test_finite(self):
        field = simple_field()
        for d in [1, 3, 7]:
            r = tau_function(field, d)
            assert math.isfinite(r.real) and math.isfinite(r.imag)

    def test_empty_field_returns_zero(self):
        field = RelationalField()
        r = tau_function(field, d=3)
        assert r == 0j
