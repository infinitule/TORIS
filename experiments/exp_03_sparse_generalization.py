"""Experiment 03 — Sparse Generalization.

Failure mode (CLAUDE.md §1.4 / §7.3): can the system reason from sparse evidence
across a long relational chain, reaching a conclusion with *calibrated*
uncertainty rather than a confident hallucination?

HYPOTHESIS
    From 4 seed relators, TORIS reaches an 8-hop conclusion (c0 → c8) by
    instantiating hypothetical connectors (σ = 0.1) for the 4 missing hops
    (MATH_SPEC §6). The conclusion does NOT collapse to zero — the chain is
    unbroken — and its strength obeys the §6 lower bound
    σ_conclusion ≥ (min_σ)^K · exp(−λ·Σ ΔS_i). Because four hops are guessed,
    the reported confidence is very low: the system knows how uncertain it is.

SETUP
    Seeds (given): c0→c1 (0.90), c2→c3 (0.80), c4→c5 (0.85), c6→c7 (0.75).
    Target route : c0 → c1 → … → c8  (K = 8 hops).
    Missing hops : c1→c2, c3→c4, c5→c6, c7→c8  → instantiated as hypotheticals.

TRANSFORMER BASELINE
    A transformer asked to bridge c0→c8 with no path in context either declines
    or hallucinates a confident answer; it has no built-in, structurally-grounded
    uncertainty for "I guessed 4 of the 8 links." TORIS attaches a calibrated
    σ_conclusion derived from the chain itself.

PASS CRITERION
    8-hop chain realized (not broken); exactly 4 hypotheticals (σ = 0.1);
    σ_chain ≥ σ_bound > 0 (graceful degradation, §6); and σ_chain is far below a
    fully-seeded baseline (uncertainty is calibrated to the guessing).
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.constants import LAMBDA, SIGMA_HYPOTHETICAL
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.chain import ReasoningChain

T = RelationType
N_HOPS = 8


def _concepts():
    return {f"c{i}": ConceptState(id=f"c{i}") for i in range(N_HOPS + 1)}


def _seeded_field(concepts, seeds):
    f = RelationalField()
    for s, t, sigma in seeds:
        f.add_relator(Relator(T.CAUSAL, concepts[s], concepts[t], sigma=sigma))
    return f


def run_experiment(verbose: bool = True) -> dict:
    concepts = _concepts()
    seeds = [
        ("c0", "c1", 0.90),
        ("c2", "c3", 0.80),
        ("c4", "c5", 0.85),
        ("c6", "c7", 0.75),
    ]
    field = _seeded_field(concepts, seeds)
    route = [concepts[f"c{i}"] for i in range(N_HOPS + 1)]

    chain = ReasoningChain()
    result = chain.infer_along(field, route)

    # calibration reference: the same route fully seeded with strong links
    full_concepts = _concepts()
    full_field = _seeded_field(
        full_concepts,
        [(f"c{i}", f"c{i+1}", 0.9) for i in range(N_HOPS)],
    )
    baseline = chain.infer_along(
        full_field, [full_concepts[f"c{i}"] for i in range(N_HOPS + 1)]
    )

    passed = (
        result.hops == N_HOPS
        and result.n_seed == 4
        and result.n_hypothetical == 4
        and not result.broken
        and result.sigma_chain >= result.sigma_bound > 0.0
        and result.sigma_chain < baseline.sigma_chain
    )

    if verbose:
        _print_report(result, baseline, passed)

    return {
        "result": result,
        "baseline_sigma_chain": baseline.sigma_chain,
        "passed": passed,
    }


def _print_report(result, baseline, passed):
    print("=" * 68)
    print("TORIS Experiment 03 — Sparse Generalization")
    print("=" * 68)
    print(f"Seeds (given)            : {result.n_seed}")
    print(f"Hops to conclusion (K)   : {result.hops}")
    print(
        f"Hypothetical connectors  : {result.n_hypothetical} "
        f"(σ = {SIGMA_HYPOTHETICAL} each)"
    )
    print(f"Chain broken?            : {result.broken}")
    print("-" * 68)
    print("Path c0 → c8:")
    for r in result.path:
        kind = "seed " if r.sigma > SIGMA_HYPOTHETICAL else "HYPOTH"
        print(
            f"  [{kind}] {r.src_id}→{r.tgt_id}  {r.tau.name}  "
            f"σ={r.sigma:.2f}  ε={r.epsilon:.2f}"
        )
    print("-" * 68)
    print(f"min_σ along chain        : {result.min_sigma:.3f}")
    print(f"Σ ΔS (accumulated surprise): {result.sum_surprise:.3f}")
    print(f"σ_chain  (realized Π σ_i)  : {result.sigma_chain:.3e}")
    print(
        f"σ_bound  (§6 lower bound)  : {result.sigma_bound:.3e}  "
        f"[(min_σ)^K · exp(−{LAMBDA}·ΣΔS)]"
    )
    print(f"§6 holds (σ_chain ≥ bound) : {result.sigma_chain >= result.sigma_bound}")
    print("-" * 68)
    print(
        f"Calibration — fully-seeded baseline σ_chain: " f"{baseline.sigma_chain:.3e}"
    )
    print(
        f"  guessing 4/8 links lowers confidence by "
        f"{baseline.sigma_chain / result.sigma_chain:.0f}×"
    )
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(f"VERDICT: {verdict}  (8-hop conclusion, unbroken, calibrated)")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
