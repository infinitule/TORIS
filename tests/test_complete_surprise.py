"""Tests for Layer 9 — UnifiedSurprise (complete_surprise.py)."""
import math
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.complete_surprise import UnifiedSurprise, UnifiedResult, _is_suppressed


def simple_field(n=5):
    field = RelationalField()
    taus = list(RelationType)
    cs = [ConceptState(id=f"U{i}") for i in range(n + 1)]
    for c in cs: field.add_concept(c)
    for i in range(n):
        field.add_relator(Relator(
            tau=taus[i % len(taus)], src=cs[i], tgt=cs[i+1],
            sigma=0.6, kappa=0.5, epsilon=0.3
        ))
    return field


class TestIsSupressed:
    def test_d4_suppressed(self):
        assert _is_suppressed(4)   # 4 mod 5 == 4

    def test_d5_suppressed(self):
        assert _is_suppressed(5)   # 5 mod 7 == 5

    def test_d1_not_suppressed(self):
        assert not _is_suppressed(1)

    def test_d3_not_suppressed(self):
        assert not _is_suppressed(3)

    def test_d7_not_suppressed(self):
        # 7%5=2, 7%7=0, 7%11=7 — none match suppression conditions
        assert not _is_suppressed(7)


class TestUnifiedSurprise:
    def setup_method(self):
        self.field = simple_field()
        self.us = UnifiedSurprise(d_crit=5)

    def test_returns_unified_result(self):
        r = self.us.compute(self.field, self.field, d=3)
        assert isinstance(r, UnifiedResult)

    def test_suppressed_depth_returns_zero(self):
        r = self.us.compute(self.field, self.field, d=4)  # suppressed
        assert r.suppressed is True
        assert r.delta_S == 0.0
        assert r.regime_used == "suppressed"

    def test_shallow_uses_fast_or_standard(self):
        for d in [1, 2, 3]:
            r = self.us.compute(self.field, self.field, d=d)
            if not r.suppressed:
                assert r.regime_used in ("fast", "standard")

    def test_deep_uses_deep_regime(self):
        r = self.us.compute(self.field, self.field, d=7)
        assert r.regime_used == "deep"

    def test_all_results_finite(self):
        for d in [1, 2, 3, 6, 7, 8, 9, 10]:
            r = self.us.compute(self.field, self.field, d=d)
            assert math.isfinite(r.delta_S)

    def test_weights_sum_to_one(self):
        for d in [1, 3, 6, 8]:
            r = self.us.compute(self.field, self.field, d=d)
            if not r.suppressed:
                total = r.weights_alpha + r.weights_beta + r.weights_gamma
                assert abs(total - 1.0) < 1e-9

    def test_deep_has_error_bound(self):
        r = self.us.compute(self.field, self.field, d=7)
        assert r.error_bound >= 0

    def test_rademacher_terms_reported_for_deep(self):
        r = self.us.compute(self.field, self.field, d=7)
        assert r.rademacher_terms_used > 0

    def test_none_pred_accepted(self):
        r = self.us.compute(self.field, None, d=3)
        assert math.isfinite(r.delta_S)
