"""Tests for the full inference loop tying all layers together (§3, §5)."""

import pytest

from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.inference import InferenceLoop


def _r(s, t, tau=RelationType.CAUSAL, sigma=0.8, kappa=0.3):
    return Relator(
        tau, ConceptState(id=s), ConceptState(id=t), sigma=sigma, kappa=kappa
    )


def test_step_returns_trace_and_grows_field():
    field = RelationalField()
    field.add_relator(_r("a", "b"))
    loop = InferenceLoop(field, GoalManifold(Goal("g")))  # unconstrained

    rec = loop.step([_r("c", "d")])  # a surprising new relation
    assert rec.t == 1
    assert rec.n_added == 1
    assert loop.field.num_relators() == 2


def test_run_chain_produces_structural_drift():
    field = RelationalField()
    field.add_relator(_r("n0", "n1"))
    loop = InferenceLoop(field, GoalManifold(Goal("g")))

    # 10-step chain, each step discovers a new relation
    obs = [[_r(f"n{i}", f"n{i+1}")] for i in range(1, 11)]
    loop.run(obs)

    assert loop.field.num_relators() >= 10  # field grew
    drift = loop.drift()
    assert drift["d_struct"] > 0.0  # topology changed
    assert drift["d_topo"] > 0.0


def test_drift_zero_when_nothing_observed():
    field = RelationalField()
    field.add_relator(_r("a", "b", kappa=0.3))
    loop = InferenceLoop(field, GoalManifold(Goal("g")))
    # observe nothing surprising; no structural change
    loop.run([[] for _ in range(3)])
    drift = loop.drift()
    assert drift["d_struct"] == pytest.approx(0.0)
