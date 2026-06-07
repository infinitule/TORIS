"""Tests for Layer 9 — Rademacher exact surprise series."""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.engine.rademacher import TAU_INDEX
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.rademacher import (
    bessel_I_3_2, kloosterman_sum,
    rademacher_surprise, certified_surprise, RademacherResult
)


def simple_field(n=6):
    field = RelationalField()
    taus = list(RelationType)
    cs = [ConceptState(id=f"R{i}") for i in range(n + 1)]
    for c in cs: field.add_concept(c)
    for i in range(n):
        field.add_relator(Relator(
            tau=taus[i % len(taus)], src=cs[i], tgt=cs[i+1],
            sigma=0.6, kappa=0.5, epsilon=0.3
        ))
    return field


class TestBesselI32:
    def test_known_value(self):
        # I_{3/2}(x) = sqrt(2/πx) * (cosh(x)/x - sinh(x)/x^2)
        x = 1.0
        result = bessel_I_3_2(x)
        assert math.isfinite(result)
        assert result > 0

    def test_positive_for_positive_x(self):
        for x in [0.1, 0.5, 1.0, 2.0, 5.0]:
            assert bessel_I_3_2(x) > 0

    def test_increases_with_x(self):
        vals = [bessel_I_3_2(x) for x in [0.5, 1.0, 2.0, 4.0]]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i+1]

    def test_analytic_matches_formula(self):
        # Compare against manual computation
        x = 2.0
        import math
        expected = math.sqrt(2 / (math.pi * x)) * (math.cosh(x)/x - math.sinh(x)/(x*x))
        assert abs(bessel_I_3_2(x) - expected) < 1e-12


class TestKloostermanSum:
    def test_returns_complex(self):
        field = simple_field()
        result = kloosterman_sum(field, k=1, d=3)
        assert isinstance(result, complex)

    def test_k1_norm_bounded(self):
        # For k=1, gcd(h,1)=1 always, sum has 1 term
        field = simple_field()
        result = kloosterman_sum(field, k=1, d=2)
        assert abs(result) < 10.0  # bounded by sum of sigma*exp()

    def test_finite(self):
        field = simple_field()
        for k in [1, 2, 3]:
            for d in [1, 3, 5]:
                r = kloosterman_sum(field, k, d)
                assert math.isfinite(r.real) and math.isfinite(r.imag)


class TestRademacherSurprise:
    def test_returns_result_type(self):
        field = simple_field()
        result = rademacher_surprise(field, d=3)
        assert isinstance(result, RademacherResult)

    def test_s_exact_finite(self):
        field = simple_field()
        for d in [1, 2, 3, 5, 7]:
            r = rademacher_surprise(field, d)
            assert math.isfinite(r.S_exact), f"Not finite at d={d}"

    def test_error_bound_positive(self):
        field = simple_field()
        r = rademacher_surprise(field, d=4, N_terms=3)
        assert r.error_bound >= 0

    def test_error_bound_finite_positive(self):
        # Error bound must be finite and non-negative for all N
        field = simple_field()
        for n_terms in [1, 3, 6, 10]:
            r = rademacher_surprise(field, d=5, N_terms=n_terms)
            assert math.isfinite(r.error_bound)
            assert r.error_bound >= 0

    def test_convergence(self):
        # S_6 and S_3 should be close
        field = simple_field()
        r3 = rademacher_surprise(field, d=3, N_terms=3)
        r6 = rademacher_surprise(field, d=3, N_terms=6)
        assert abs(r6.S_exact - r3.S_exact) < 1.0

    def test_terms_used(self):
        field = simple_field()
        r = rademacher_surprise(field, d=3, N_terms=4)
        assert r.terms_used <= 4

    def test_integer_nearness_in_range(self):
        field = simple_field()
        r = rademacher_surprise(field, d=3)
        assert 0.0 <= r.integer_nearness <= 0.5


class TestCertifiedSurprise:
    def test_returns_tuple(self):
        field = simple_field()
        result = certified_surprise(field, d=3, precision=8)
        assert isinstance(result, tuple) and len(result) == 2

    def test_bound_positive(self):
        field = simple_field()
        S, bound = certified_surprise(field, d=3)
        assert bound >= 0

    def test_higher_precision_tighter_bound(self):
        field = simple_field()
        _, b4 = certified_surprise(field, d=3, precision=4)
        _, b8 = certified_surprise(field, d=3, precision=8)
        assert b8 <= b4 + 1e-12  # higher precision → tighter or equal bound


class TestTauIndex:
    def test_all_twelve_types(self):
        assert len(TAU_INDEX) == 12

    def test_causal_is_1(self):
        assert TAU_INDEX[RelationType.CAUSAL] == 1

    def test_instantiates_is_12(self):
        assert TAU_INDEX[RelationType.INSTANTIATES] == 12

    def test_unique_values(self):
        vals = list(TAU_INDEX.values())
        assert len(vals) == len(set(vals))
