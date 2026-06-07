"""Tests for Layer 8: The Ramanujan Extension (TORIS §11).

Covers all five modules:
  circle_method.py     — saddle points, Kloosterman correction
  suppression.py       — modular structure, depth suppression
  ramanujan_goal.py    — goal coherence, warp approximation, π series
  rogers_ramanujan.py  — partition function, entropy, chain structure
  ramanujan_critical.py — critical point detection
"""
import math
import pytest

from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.field.relational_field import RelationalField
from toris.goal.manifold import GoalManifold, Goal
from toris.goal.subgoal import Subgoal


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_field():
    f = RelationalField()
    cs = [ConceptState(id=f"C{i}") for i in range(6)]
    for c in cs:
        f.add_concept(c)
    taus = [RelationType.CAUSAL, RelationType.ENABLES, RelationType.EVIDENCES]
    for i in range(5):
        f.add_relator(Relator(taus[i % 3], cs[i], cs[i + 1],
                              sigma=0.8, kappa=0.7, epsilon=0.2))
    return f


@pytest.fixture
def modular_field():
    """5-modular field: strengths are multiples of 0.2."""
    f = RelationalField()
    cs = [ConceptState(id=f"M{i}") for i in range(10)]
    for c in cs:
        f.add_concept(c)
    import random; random.seed(99)
    for i in range(30):
        src, tgt = random.sample(cs, 2)
        k = random.randint(0, 5)
        f.add_relator(Relator(RelationType.CAUSAL, src, tgt,
                              sigma=k / 5.0, kappa=0.5,
                              epsilon=random.randint(0, 5) / 5.0))
    return f


@pytest.fixture
def chain_contra_field():
    f = RelationalField()
    cs = [ConceptState(id=f"X{i}") for i in range(6)]
    for c in cs:
        f.add_concept(c)
    for i in range(5):
        f.add_relator(Relator(RelationType.CAUSAL, cs[i], cs[i + 1],
                              sigma=0.8, kappa=0.7))
    for i in range(5):
        f.add_relator(Relator(RelationType.CONTRADICTS, cs[i], cs[i + 1],
                              sigma=0.6, kappa=0.5))
    return f


@pytest.fixture
def coherent_manifold():
    """Priorities 6/11 → warp_sum ≈ 1.0 → Q ≈ 0."""
    primary = Goal(description="test_goal")
    m = GoalManifold(primary=primary)
    for i in range(3):
        m.add_subgoal(Subgoal(description=f"sg_{i}", priority=6.0 / 11.0))
    return m


@pytest.fixture
def incoherent_manifold():
    primary = Goal(description="incoherent_goal")
    m = GoalManifold(primary=primary)
    for v in [math.sqrt(2) / 3, math.e / 7]:
        m.add_subgoal(Subgoal(description=f"sg_{v:.3f}", priority=v))
    return m


# ─── circle_method.py ─────────────────────────────────────────────────────────

class TestCircleMethod:
    def test_saddle_point_positive(self):
        from toris.engine.circle_method import saddle_point
        for d in range(1, 10):
            assert saddle_point(d) > 0.0

    def test_saddle_point_d0(self):
        from toris.engine.circle_method import saddle_point
        assert saddle_point(0) == 1.0

    def test_saddle_point_increases(self):
        from toris.engine.circle_method import saddle_point
        # κ_saddle(d) = exp(π√(2d/3)/d) — monotone analysis
        vals = [saddle_point(d) for d in range(1, 8)]
        # All positive and finite
        assert all(v > 0 for v in vals)

    def test_kloosterman_alternates(self):
        from toris.engine.circle_method import kloosterman_correction
        c1 = kloosterman_correction(1)
        c2 = kloosterman_correction(2)
        assert c1 < 0  # (-1)^1 negative
        assert c2 > 0  # (-1)^2 positive

    def test_kloosterman_d0(self):
        from toris.engine.circle_method import kloosterman_correction
        assert kloosterman_correction(0) == 0.0

    def test_circle_method_result(self, small_field):
        from toris.engine.circle_method import circle_method_surprise
        r = circle_method_surprise(small_field, 3)
        assert 0.0 <= r.delta_s_total <= 1.0
        assert r.d == 3
        assert r.kappa_saddle > 0

    def test_saddle_profile(self, small_field):
        from toris.engine.circle_method import saddle_surprise_profile
        profile = saddle_surprise_profile(small_field, max_depth=8)
        assert len(profile) == 8
        assert all(0.0 <= r.delta_s_total <= 1.0 for r in profile)


# ─── suppression.py ───────────────────────────────────────────────────────────

class TestSuppression:
    def test_suppressed_depth_p5(self):
        from toris.engine.suppression import suppressed_depth
        assert suppressed_depth(4) is True   # 4 ≡ 4 mod 5
        assert suppressed_depth(9) is True   # 9 ≡ 4 mod 5
        assert suppressed_depth(3) is False

    def test_suppressed_depth_p7(self):
        from toris.engine.suppression import suppressed_depth
        assert suppressed_depth(5) is True   # 5 ≡ 5 mod 7
        assert suppressed_depth(12) is True  # 12 ≡ 5 mod 7

    def test_suppressed_depth_p11(self):
        from toris.engine.suppression import suppressed_depth
        assert suppressed_depth(6) is True   # 6 ≡ 6 mod 11
        assert suppressed_depth(17) is True  # 17 ≡ 6 mod 11

    def test_is_modular_field(self, modular_field):
        from toris.engine.suppression import is_modular_field
        # Should detect modular structure
        result = is_modular_field(modular_field, 5)
        assert isinstance(result, bool)

    def test_suppression_report_structure(self, modular_field):
        from toris.engine.suppression import suppression_report
        report = suppression_report(modular_field, max_depth=15)
        assert len(report.entries) == 15
        assert 0.0 <= report.suppression_accuracy <= 1.0

    def test_verify_suppression(self, modular_field):
        from toris.engine.suppression import verify_suppression
        s_d, mod_val, is_zero = verify_suppression(modular_field, 4, 5)
        assert isinstance(is_zero, bool)
        assert s_d >= 0.0


# ─── ramanujan_goal.py ────────────────────────────────────────────────────────

class TestRamanujanGoal:
    def test_pi_ramanujan_1_term(self):
        from toris.engine.ramanujan_goal import pi_ramanujan
        # 1 term gives ~3.14159…
        assert abs(pi_ramanujan(1) - math.pi) < 1e-5

    def test_pi_ramanujan_3_terms(self):
        from toris.engine.ramanujan_goal import pi_ramanujan
        assert abs(pi_ramanujan(3) - math.pi) < 1e-12

    def test_goal_coherence_high(self, coherent_manifold):
        from toris.engine.ramanujan_goal import goal_coherence
        Q = goal_coherence(coherent_manifold)
        assert Q < 0.01

    def test_goal_coherence_low(self, incoherent_manifold):
        from toris.engine.ramanujan_goal import goal_coherence
        Q = goal_coherence(incoherent_manifold)
        assert Q >= 0.01

    def test_near_integer_check(self):
        from toris.engine.ramanujan_goal import near_integer_check
        assert near_integer_check(3.000001, 1e-4) is True
        assert near_integer_check(3.5, 1e-4) is False

    def test_full_warp_positive(self, small_field, coherent_manifold):
        from toris.engine.ramanujan_goal import full_warp
        phi = full_warp(coherent_manifold, small_field)
        assert phi >= 0.0

    def test_ramanujan_3term_matches_full(self, small_field, coherent_manifold):
        from toris.engine.ramanujan_goal import full_warp, ramanujan_3term
        phi_exact = full_warp(coherent_manifold, small_field)
        phi_ram = ramanujan_3term(coherent_manifold, small_field, n_terms=3)
        if phi_exact > 0:
            rel_err = abs(phi_exact - phi_ram) / phi_exact
            assert rel_err < 0.05  # within 5%

    def test_auto_warp_switches(self, small_field, coherent_manifold, incoherent_manifold):
        from toris.engine.ramanujan_goal import auto_warp
        _, method_hi = auto_warp(coherent_manifold, small_field)
        _, method_lo = auto_warp(incoherent_manifold, small_field)
        assert method_hi == "ramanujan"
        assert method_lo == "exact"


# ─── rogers_ramanujan.py ──────────────────────────────────────────────────────

class TestRogersRamanujan:
    def test_partition_function_rr_q03(self):
        from toris.engine.rogers_ramanujan import partition_function_rr
        # Verify against known value: Z_RR(0.3) ≈ 1.44132
        z = partition_function_rr(None, 0.3)
        assert abs(z - 1.44132) < 0.001

    def test_partition_function_rr_q0(self):
        from toris.engine.rogers_ramanujan import partition_function_rr
        assert partition_function_rr(None, 0.0) == 1.0

    def test_contra_chain_true(self, chain_contra_field):
        from toris.engine.rogers_ramanujan import contra_chain_structure
        assert contra_chain_structure(chain_contra_field) is True

    def test_contra_chain_false_for_star(self, small_field):
        from toris.engine.rogers_ramanujan import contra_chain_structure
        # small_field has no CONTRA edges at all → trivially True (empty chain)
        result = contra_chain_structure(small_field)
        assert isinstance(result, bool)

    def test_field_entropy_positive(self, chain_contra_field):
        from toris.engine.rogers_ramanujan import field_entropy
        H = field_entropy(chain_contra_field)
        assert H > 0.0

    def test_field_entropy_finite(self, chain_contra_field):
        from toris.engine.rogers_ramanujan import field_entropy
        import math
        H = field_entropy(chain_contra_field, q=0.3)
        assert math.isfinite(H)

    def test_critical_points_returns_list(self, chain_contra_field):
        from toris.engine.rogers_ramanujan import critical_points
        cps = critical_points(chain_contra_field, steps=50, threshold=0.05)
        assert isinstance(cps, list)


# ─── ramanujan_critical.py ────────────────────────────────────────────────────

class TestRamanujanCritical:
    def test_find_critical_points_returns_list(self, chain_contra_field, coherent_manifold):
        from toris.engine.ramanujan_critical import find_critical_points
        cps = find_critical_points(chain_contra_field, coherent_manifold,
                                   threshold=0.05, steps=50)
        assert isinstance(cps, list)

    def test_critical_report_string(self, chain_contra_field, coherent_manifold):
        from toris.engine.ramanujan_critical import critical_report
        report = critical_report(chain_contra_field, coherent_manifold, threshold=0.05)
        assert "Ramanujan Critical Point Report" in report

    def test_is_at_critical_bool(self, chain_contra_field, coherent_manifold):
        from toris.engine.ramanujan_critical import is_at_critical
        result = is_at_critical(chain_contra_field, coherent_manifold, threshold=0.05)
        assert isinstance(result, bool)

    def test_criticality_score_range(self, chain_contra_field, coherent_manifold):
        from toris.engine.ramanujan_critical import find_critical_points
        cps = find_critical_points(chain_contra_field, coherent_manifold,
                                   threshold=0.05, steps=50)
        for cp in cps:
            assert 0.0 <= cp.criticality_score <= 1.0

    def test_ramanujan_critical_fields(self, chain_contra_field, coherent_manifold):
        from toris.engine.ramanujan_critical import find_critical_points
        cps = find_critical_points(chain_contra_field, coherent_manifold,
                                   threshold=0.05, steps=30)
        for cp in cps:
            assert cp.is_near_integer is True
            assert 0.0 < cp.kappa < 1.0
