"""Experiment 06 — Cross-Domain Generalization via Medium-Plasticity Consolidation.

New failure mode (beyond the original five): can TORIS *generalize to a new
domain* after consolidating a relational schema across several sessions
(MATH_SPEC §5.2)?

HYPOTHESIS
    TORIS represents structure at the level of typed relators over relational
    roles — not domain-specific coordinates. So a relational *schema* consolidated
    in one domain should transfer to a structurally-analogous new domain. After 5
    sessions of medium-plasticity consolidation (§5.2) on an abstract causal-chain
    schema (r0 →CAUSAL→ r1 →CAUSAL→ r2), a brand-new domain B that instantiates
    the same schema reaches its conclusion with materially higher (and useful)
    confidence than a cold, unconsolidated model.

SETUP
    Schema (abstract roles): r0→r1, r1→r2, CAUSAL, weak prior σ₀ = 0.2.
    Domain-A sessions (×5): each session presents a *novel* scenario whose
        structure the schema must (surprisingly) account for. The schema is
        therefore invoked with structural surprise (ε = α) each session; medium
        plasticity moves its baseline σ toward that surprise level (§5.2). Slow
        plasticity accrues a long-term baseline in parallel.
    Domain B (new concepts b0,b1,b2): instantiate the SAME schema and read off
        the b0 → b2 conclusion strength, using the (transferred) schema σ.

TRANSFORMER BASELINE
    Cross-domain transfer in a transformer requires gradient training on the new
    distribution; at inference it cannot consolidate a reusable schema from a few
    episodes. The "cold" arm here (no consolidation) stands in for the
    un-adapted model.

PASS CRITERION
    (1) schema σ increases monotonically across the 5 sessions (consolidation);
    (2) consolidated domain-B conclusion > cold conclusion (transfer);
    (3) consolidated crosses the usefulness threshold τ that cold does not.
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.constants import ETA_MED
from toris.engine.predictive import PredictiveEngine
from toris.field.relational_field import RelationalField
from toris.plasticity.medium import MediumPlasticity
from toris.plasticity.slow import SlowPlasticity
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.chain import ReasoningChain

T = RelationType
SCHEMA_PRIOR = 0.2
SESSIONS = 5
USEFUL_THRESHOLD = 0.05  # min conclusion strength to "confidently generalize"


def build_schema(prior: float):
    roles = {f"r{i}": ConceptState(id=f"r{i}") for i in range(3)}
    f = RelationalField()
    f.add_relator(Relator(T.CAUSAL, roles["r0"], roles["r1"], sigma=prior))
    f.add_relator(Relator(T.CAUSAL, roles["r1"], roles["r2"], sigma=prior))
    return f, roles


def schema_sigmas(field, roles):
    return [
        field.relators_between(roles["r0"], roles["r1"])[0].sigma,
        field.relators_between(roles["r1"], roles["r2"])[0].sigma,
    ]


def session_surprise(engine, schema):
    """A novel domain scenario invokes the schema → structural surprise (ε = α).

    Modeled as a cold prediction (the new scenario was not anticipated) against
    the observed schema structure, so each schema relator is unpredicted and
    carries ε = α (MATH_SPEC §3.3). Computed by the real surprise metric.
    """
    pred = RelationalField()  # cold: the novel scenario was not predicted
    obs = RelationalField()
    obs.add_relators([r.clone() for r in schema.relators()])  # rids preserved
    return engine.compute_delta(pred, obs)


def domain_b_conclusion(sigmas):
    """Instantiate the schema on fresh domain-B concepts and compose the chain."""
    b = {f"b{i}": ConceptState(id=f"b{i}") for i in range(3)}
    path = [
        Relator(T.CAUSAL, b["b0"], b["b1"], sigma=sigmas[0]),
        Relator(T.CAUSAL, b["b1"], b["b2"], sigma=sigmas[1]),
    ]
    composite = ReasoningChain().compose_path(path)
    return composite.sigma  # realized Π σ_i for the b0 → b2 conclusion


def run_experiment(verbose: bool = True) -> dict:
    engine = PredictiveEngine()
    medium = MediumPlasticity(eta_med=ETA_MED)
    slow = SlowPlasticity()

    schema, roles = build_schema(SCHEMA_PRIOR)
    trajectory = [schema_sigmas(schema, roles)]  # σ after each session
    slow_trajectory = []

    for _ in range(SESSIONS):
        report = session_surprise(engine, schema)
        medium.observe_report(report)
        medium.consolidate(schema)  # §5.2 update
        slow.consolidate(schema)  # long-term baseline EMA
        trajectory.append(schema_sigmas(schema, roles))
        slow_trajectory.append([slow.baseline_of(r.rid) for r in schema.relators()])

    consolidated_sigmas = schema_sigmas(schema, roles)
    cold = domain_b_conclusion([SCHEMA_PRIOR, SCHEMA_PRIOR])
    consolidated = domain_b_conclusion(consolidated_sigmas)

    # (1) monotonic per-relator increase across sessions
    monotonic = all(
        trajectory[s + 1][j] > trajectory[s][j]
        for s in range(SESSIONS)
        for j in range(len(consolidated_sigmas))
    )
    passed = (
        monotonic
        and consolidated > cold
        and consolidated > USEFUL_THRESHOLD
        and cold < USEFUL_THRESHOLD
    )

    if verbose:
        _print_report(trajectory, slow_trajectory, cold, consolidated, passed)

    return {
        "trajectory": trajectory,
        "slow_baseline_final": slow_trajectory[-1] if slow_trajectory else [],
        "cold_conclusion": cold,
        "consolidated_conclusion": consolidated,
        "lift": consolidated / cold if cold else float("inf"),
        "monotonic": monotonic,
        "passed": passed,
    }


def _print_report(trajectory, slow_trajectory, cold, consolidated, passed):
    print("=" * 68)
    print("TORIS Experiment 06 — Cross-Domain Generalization (Medium Plasticity)")
    print("=" * 68)
    print(
        f"Schema: r0 →CAUSAL→ r1 →CAUSAL→ r2   (prior σ = {SCHEMA_PRIOR}, "
        f"η_med = {ETA_MED})"
    )
    print("-" * 68)
    print("Schema σ consolidation across sessions (§5.2):")
    for s, sig in enumerate(trajectory):
        tag = "prior" if s == 0 else f"sess {s}"
        print(f"  {tag:>7}:  σ(r0→r1)={sig[0]:.4f}   σ(r1→r2)={sig[1]:.4f}")
    print(
        f"  slow long-term baseline (final): "
        f"{[round(b, 4) for b in slow_trajectory[-1]]}"
    )
    print("-" * 68)
    print("Domain B (new concepts b0,b1,b2) — same schema instantiated:")
    print(f"  cold model        b0→b2 conclusion σ = {cold:.4f}")
    print(f"  consolidated model b0→b2 conclusion σ = {consolidated:.4f}")
    print(f"  transfer lift                       = {consolidated / cold:.2f}×")
    print(f"  usefulness threshold τ              = {USEFUL_THRESHOLD}")
    cold_ok = "below τ → cannot generalize" if cold < USEFUL_THRESHOLD else "above τ"
    cons_ok = "above τ → generalizes" if consolidated > USEFUL_THRESHOLD else "below τ"
    print(f"  cold        : {cold_ok}")
    print(f"  consolidated : {cons_ok}")
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(
        f"VERDICT: {verdict}  (monotonic consolidation + threshold-crossing transfer)"
    )
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
