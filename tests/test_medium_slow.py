"""Tests for medium (§5.2) and slow (training-analog) plasticity."""

import pytest

from toris.field.relational_field import RelationalField
from toris.plasticity.medium import MediumPlasticity
from toris.plasticity.slow import SlowPlasticity
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


def _field(*relators):
    f = RelationalField()
    f.add_relators(relators)
    return f


def _r(s, t, sigma):
    return Relator(
        RelationType.CAUSAL, ConceptState(id=s), ConceptState(id=t), sigma=sigma
    )


# -- medium plasticity (MATH_SPEC §5.2) -------------------------------------


def test_medium_consolidate_matches_formula():
    r = _r("a", "b", 0.2)
    field = _field(r)
    med = MediumPlasticity(eta_med=0.05)
    med.observe(r, 0.6)  # session surprise level = 0.6
    med.consolidate(field)
    # σ' = 0.2 + 0.05·(0.6 − 0.2) = 0.22
    assert r.sigma == pytest.approx(0.22)


def test_medium_surprise_level_is_session_mean():
    r = _r("a", "b", 0.5)
    med = MediumPlasticity()
    med.observe(r, 0.6)
    med.observe(r, 0.4)
    assert med.surprise_level(r.rid) == pytest.approx(0.5)  # mean of 0.6, 0.4


def test_medium_unobserved_relator_fades():
    r = _r("a", "b", 0.8)
    field = _field(r)
    med = MediumPlasticity(eta_med=0.05)
    med.consolidate(field)  # never observed → target 0 → fades
    # σ' = 0.8 + 0.05·(0 − 0.8) = 0.76
    assert r.sigma == pytest.approx(0.76)


def test_medium_consolidation_is_monotonic_toward_target():
    r = _r("a", "b", 0.2)
    field = _field(r)
    med = MediumPlasticity(eta_med=0.05)
    sigmas = [r.sigma]
    for _ in range(5):
        med.observe(r, 0.6)
        med.consolidate(field)
        sigmas.append(r.sigma)
    # strictly increasing, and approaching (never overshooting) the target 0.6
    assert all(b > a for a, b in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] < 0.6
    assert med.session_history and len(med.session_history) == 5


def test_medium_session_reset():
    r = _r("a", "b", 0.5)
    field = _field(r)
    med = MediumPlasticity()
    med.observe(r, 0.9)
    med.consolidate(field)
    # accumulators cleared → next session with no observation targets 0
    assert med.surprise_level(r.rid) == 0.0


# -- slow plasticity (training analog) --------------------------------------


def test_slow_baseline_ema_and_consolidation():
    r = _r("a", "b", 0.6)
    field = _field(r)
    slow = SlowPlasticity(eta_slow=0.5, consolidation_threshold=0.4)
    slow.consolidate(field)
    # baseline init = σ = 0.6, EMA toward 0.6 stays 0.6 ; 0.6 ≥ 0.4 → consolidated
    assert slow.baseline_of(r.rid) == pytest.approx(0.6)
    assert slow.is_consolidated(r.rid)


def test_slow_floor_protects_consolidated_relator():
    r = _r("a", "b", 0.6)
    field = _field(r)
    slow = SlowPlasticity(eta_slow=0.5, consolidation_threshold=0.4)
    slow.consolidate(field)
    r.sigma = 0.1  # simulate aggressive fast decay
    slow.apply_floor(field)
    assert r.sigma == pytest.approx(0.6)  # restored to protected baseline


def test_slow_does_not_protect_unconsolidated():
    r = _r("a", "b", 0.2)
    field = _field(r)
    slow = SlowPlasticity(eta_slow=0.5, consolidation_threshold=0.9)
    slow.consolidate(field)  # baseline 0.2 < 0.9 → not consolidated
    r.sigma = 0.05
    slow.apply_floor(field)
    assert r.sigma == pytest.approx(0.05)  # left untouched
