"""Experiment 05 — Structural Drift.

Failure mode (CLAUDE.md §1.4 / §7.5): does the system measurably RESTRUCTURE
itself over a reasoning chain — is F^T topologically different from F^0?

HYPOTHESIS
    Running a 50-step reasoning chain through the full inference loop (predictive
    coding + fast plasticity, MATH_SPEC §5.1) leaves the field measurably
    changed: d_topo(F^0, F^T) > 0.1 (MATH_SPEC §5.3). Inference is not reading a
    static structure — it is rewriting a live one.

SETUP
    F^0 : 5 CAUSAL relators among concepts k0..k5 (σ=0.8, κ=0.3 so the warp keeps
          them active without amplifying — drift comes from plasticity, not Φ's
          σ-boost). Goal is unconstrained (broad reasoning) so nothing is
          suppressed and drift is purely the system's own restructuring.
    Chain: 50 steps. Each step DISCOVERS a new relation to a new concept (an
           ADD: structural growth) and RE-DERIVES two stable facts (repeated
           confirmation → WEAKEN: strength drift).

TRANSFORMER BASELINE
    Weights update only during training; at inference the parameter tensor — and
    therefore the represented structure — is frozen. d_topo over a forward pass
    is exactly 0. The model cannot restructure itself while reasoning.

PASS CRITERION
    d_topo(F^0, F^50) > 0.1.
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.inference import InferenceLoop

T = RelationType
CHAIN_LEN = 50
DRIFT_BAR = 0.1
STABLE_EDGES = [("k0", "k1"), ("k2", "k3")]


def build_initial():
    c = {f"k{i}": ConceptState(id=f"k{i}") for i in range(6)}
    f = RelationalField()
    for s, t in [("k0", "k1"), ("k2", "k3"), ("k0", "k2"), ("k1", "k4"), ("k3", "k5")]:
        f.add_relator(Relator(T.CAUSAL, c[s], c[t], sigma=0.8, kappa=0.3))
    return f, c


def run_experiment(verbose: bool = True) -> dict:
    field, concepts = build_initial()
    manifold = GoalManifold(Goal("characterize the system"))  # unconstrained
    loop = InferenceLoop(field, manifold)

    e0 = loop.initial_snapshot.n_edges
    anchors = list(concepts.values())

    for t in range(1, CHAIN_LEN + 1):
        obs = []
        # DISCOVERY: a newly inferred relation to a fresh concept (ADD)
        new = ConceptState(id=f"d{t}")
        anchor = anchors[(t - 1) % len(anchors)]
        obs.append(Relator(T.CAUSAL, anchor, new, sigma=0.7, kappa=0.3))
        anchors.append(new)
        # RE-DERIVATION: confirm stable facts repeatedly (drives WEAKEN)
        for s, tt in STABLE_EDGES:
            existing = loop.field.relators_between(s, tt)
            if existing:
                obs.append(existing[0])
        loop.step(obs)

    drift = loop.drift()
    total_added = sum(r.n_added for r in loop.history)
    total_weakened = sum(r.n_weakened for r in loop.history)
    total_strengthened = sum(r.n_strengthened for r in loop.history)
    eT = loop.field.num_relators()

    passed = drift["d_topo"] > DRIFT_BAR

    if verbose:
        _print_report(
            e0, eT, total_added, total_strengthened, total_weakened, drift, passed
        )

    return {
        "edges_t0": e0,
        "edges_tT": eT,
        "total_added": total_added,
        "total_strengthened": total_strengthened,
        "total_weakened": total_weakened,
        "drift": drift,
        "passed": passed,
    }


def _print_report(e0, eT, added, strengthened, weakened, drift, passed):
    print("=" * 68)
    print("TORIS Experiment 05 — Structural Drift")
    print("=" * 68)
    print(f"Chain length            : {CHAIN_LEN} inference steps")
    print(f"Edges at t=0            : {e0}")
    print(f"Edges at t={CHAIN_LEN}           : {eT}")
    print(
        f"Plasticity events       : {added} ADD / {strengthened} STRENGTHEN / "
        f"{weakened} WEAKEN"
    )
    print("-" * 68)
    print(f"d_struct                : {drift['d_struct']:.4f}")
    print(f"d_type                  : {drift['d_type']:.4f}")
    print(f"d_strength              : {drift['d_strength']:.4f}")
    print(f"d_topo (= mean of three): {drift['d_topo']:.4f}")
    print("Transformer baseline    : d_topo = 0.0 (frozen at inference)")
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(f"VERDICT: {verdict}  (bar: d_topo > {DRIFT_BAR})")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
