"""Tests for fast plasticity ΔF and structural drift (MATH_SPEC §5.1, §5.3)."""

import pytest

from toris.constants import THETA_EPSILON
from toris.engine.surprise import RelatorSurprise, SurpriseReport
from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.plasticity.fast import FastPlasticity, snapshot, structural_drift
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@pytest.fixture
def concepts():
    return {n: ConceptState(id=n) for n in ["A", "B", "C", "D"]}


def _r(concepts, s, t, tau=RelationType.CAUSAL, sigma=0.8, kappa=1.0):
    return Relator(tau, concepts[s], concepts[t], sigma=sigma, kappa=kappa)


def _report(entries):
    """Build a SurpriseReport from (relator, predicted, epsilon) tuples."""
    per = {}
    for relator, predicted, eps in entries:
        per[relator.rid] = RelatorSurprise(
            relator=relator,
            predicted=predicted,
            eps_struct=0.0 if predicted else 1.0,
            eps_type=0.0,
            eps_strength=0.0,
            epsilon=eps,
        )
    return SurpriseReport(0.0, 0.0, 0.0, 0.0, per, THETA_EPSILON)


# unconstrained goal keeps everything active; high θ_amplify isolates plasticity
def _open_manifold():
    return GoalManifold(Goal("g"))  # empty scope → relevance 1.0 everywhere


def test_add_inserts_surprising_unpredicted_relator(concepts):
    field = RelationalField()
    novel = _r(concepts, "C", "D")  # not in field, high surprise
    report = _report([(novel, False, 0.6)])  # ε = 0.6 > θ_add = 0.4
    fp = FastPlasticity()
    warped, delta = fp.step(field, report, _open_manifold(), theta_amplify=2.0)
    assert novel.rid in {r.rid for r in warped.relators()}
    assert delta.added and delta.added[0] is novel


def test_low_surprise_observation_not_added(concepts):
    field = RelationalField()
    novel = _r(concepts, "C", "D")
    report = _report([(novel, False, 0.3)])  # ε = 0.3 < θ_add = 0.4
    fp = FastPlasticity()
    warped, delta = fp.step(field, report, _open_manifold(), theta_amplify=2.0)
    assert novel.rid not in {r.rid for r in warped.relators()}
    assert not delta.added


def test_strengthen_increases_sigma(concepts):
    field = RelationalField()
    r = _r(concepts, "A", "B", sigma=0.5)
    field.add_relator(r)
    report = _report([(r, True, 0.4)])  # ε = 0.4 > θ_strong = 0.3
    fp = FastPlasticity()
    warped, delta = fp.step(field, report, _open_manifold(), theta_amplify=2.0)
    wr = warped.relators_between("A", "B")[0]
    # σ += η_fast·ε = 0.5 + 0.1·0.4 = 0.54
    assert wr.sigma == pytest.approx(0.54)
    assert delta.strengthened


def test_weaken_after_n_confirmations(concepts):
    field = RelationalField()
    r = _r(concepts, "A", "B", sigma=0.8)
    field.add_relator(r)
    m = _open_manifold()
    fp = FastPlasticity(confirm_n=2)

    def confirmed(fld):
        rr = fld.relators_between("A", "B")[0]
        return _report([(rr, True, 0.0)])  # confirmed: ε = 0 ≤ θ_ε

    # step 1: count → 1 (no weaken yet)
    field, d1 = fp.step(field, confirmed(field), m, theta_amplify=2.0)
    assert not d1.weakened
    assert field.relators_between("A", "B")[0].sigma == pytest.approx(0.8)
    # step 2: count → 2 ≥ N → weaken by η_decay
    field, d2 = fp.step(field, confirmed(field), m, theta_amplify=2.0)
    assert d2.weakened
    assert field.relators_between("A", "B")[0].sigma == pytest.approx(0.79)


def test_suppress_via_warp(concepts):
    field = RelationalField()
    keep = _r(concepts, "A", "B")
    drop = _r(concepts, "C", "D")
    field.add_relator(keep)
    field.add_relator(drop)
    # goal scope = {A,B} → C→D suppressed by Φ
    m = GoalManifold(Goal("g", concepts={"A", "B"}))
    report = _report([])  # no observations this step
    fp = FastPlasticity()
    warped, delta = fp.step(field, report, m)
    assert ("A", "B") in warped.edge_set()
    assert ("C", "D") not in warped.edge_set()
    assert drop.rid in {r.rid for r in delta.suppressed}


# -- structural drift (MATH_SPEC §5.3) --------------------------------------


def test_structural_drift_components(concepts):
    f0 = RelationalField()
    f0.add_relator(_r(concepts, "A", "B", sigma=0.8))
    f0.add_relator(_r(concepts, "B", "C", sigma=0.8))
    snap0 = snapshot(f0)

    ft = RelationalField()
    ft.add_relator(_r(concepts, "A", "B", sigma=0.5))  # strength changed
    ft.add_relator(_r(concepts, "B", "C", sigma=0.8))
    ft.add_relator(_r(concepts, "C", "D", sigma=0.8))  # new edge
    snapT = snapshot(ft)

    drift = structural_drift(snap0, snapT)
    # d_struct = |{(C,D)}| / max(3,2) = 1/3
    assert drift["d_struct"] == pytest.approx(1 / 3)
    # matched {(A,B),(B,C)}: d_type 0 ; d_strength = (0.3² + 0)/2 = 0.045
    assert drift["d_type"] == pytest.approx(0.0)
    assert drift["d_strength"] == pytest.approx(0.045)
    assert drift["d_topo"] == pytest.approx((1 / 3 + 0 + 0.045) / 3)


def test_drift_zero_for_identical_topology(concepts):
    f = RelationalField()
    f.add_relator(_r(concepts, "A", "B"))
    snap = snapshot(f)
    drift = structural_drift(snap, snap)
    assert drift["d_topo"] == pytest.approx(0.0)
