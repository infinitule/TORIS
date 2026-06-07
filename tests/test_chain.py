"""Tests for the ReasoningChain executor and §6 sparse generalization."""

import math

import pytest

from toris.constants import ALPHA, LAMBDA, SIGMA_HYPOTHETICAL
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.chain import ReasoningChain


@pytest.fixture
def concepts():
    return {f"c{i}": ConceptState(id=f"c{i}") for i in range(9)}


def test_instantiate_hypothetical():
    a, b = ConceptState(id="a"), ConceptState(id="b")
    hyp = ReasoningChain.instantiate_hypothetical(a, b)
    assert hyp.sigma == pytest.approx(SIGMA_HYPOTHETICAL)
    assert hyp.epsilon == pytest.approx(ALPHA)  # instantiation is a structural surprise
    assert hyp.tau == RelationType.CAUSAL


def test_compose_path_product(concepts):
    r1 = Relator(RelationType.CAUSAL, concepts["c0"], concepts["c1"], sigma=0.9)
    r2 = Relator(RelationType.CAUSAL, concepts["c1"], concepts["c2"], sigma=0.8)
    r3 = Relator(RelationType.CAUSAL, concepts["c2"], concepts["c3"], sigma=0.5)
    comp = ReasoningChain().compose_path([r1, r2, r3])
    assert comp is not None
    assert comp.edge == ("c0", "c3")
    assert comp.sigma == pytest.approx(0.9 * 0.8 * 0.5)


def test_chain_uncertainty_formula(concepts):
    chain = ReasoningChain()
    path = [
        Relator(
            RelationType.CAUSAL, concepts["c0"], concepts["c1"], sigma=0.9, epsilon=0.0
        ),
        Relator(
            RelationType.CAUSAL, concepts["c1"], concepts["c2"], sigma=0.1, epsilon=0.6
        ),
    ]
    # (min_σ)^K · exp(−λ·ΣΔS) = 0.1² · exp(−0.5·0.6)
    expected = (0.1**2) * math.exp(-LAMBDA * 0.6)
    assert chain.chain_uncertainty(path) == pytest.approx(expected)


def test_infer_along_bridges_gaps_with_hypotheticals(concepts):
    # 4 seed relators, 4 missing connectors → 8-hop chain
    field = RelationalField()
    seeds = [
        ("c0", "c1", 0.9),
        ("c2", "c3", 0.8),
        ("c4", "c5", 0.85),
        ("c6", "c7", 0.75),
    ]
    for s, t, sig in seeds:
        field.add_relator(
            Relator(RelationType.CAUSAL, concepts[s], concepts[t], sigma=sig)
        )

    route = [concepts[f"c{i}"] for i in range(9)]  # c0..c8 → 8 hops
    result = ReasoningChain().infer_along(field, route)

    assert result.hops == 8
    assert result.n_seed == 4
    assert result.n_hypothetical == 4
    assert not result.broken
    assert result.min_sigma == pytest.approx(SIGMA_HYPOTHETICAL)
    # realized strength = product of all 8 σ
    expected_chain = 0.9 * 0.8 * 0.85 * 0.75 * (0.1**4)
    assert result.sigma_chain == pytest.approx(expected_chain)
    # §6 conjecture: the conclusion meets-or-exceeds the lower bound, both > 0
    assert result.sigma_chain >= result.sigma_bound > 0.0


def test_infer_along_calibration_more_guesses_lower_confidence(concepts):
    # fully-seeded 4-hop chain is far more confident than a gap-filled one
    field = RelationalField()
    for i in range(4):
        field.add_relator(
            Relator(
                RelationType.CAUSAL, concepts[f"c{i}"], concepts[f"c{i+1}"], sigma=0.9
            )
        )
    chain = ReasoningChain()
    seeded = chain.infer_along(field, [concepts[f"c{i}"] for i in range(5)])

    empty = RelationalField()
    guessed = chain.infer_along(empty, [concepts[f"c{i}"] for i in range(5)])

    assert seeded.n_hypothetical == 0
    assert guessed.n_hypothetical == 4
    assert seeded.sigma_chain > guessed.sigma_chain  # calibrated to uncertainty
