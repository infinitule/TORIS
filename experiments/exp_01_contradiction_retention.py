"""Experiment 01 — Contradiction Retention.

Failure mode (CLAUDE.md §1.4 / §7.1): does the system HOLD unresolved tension
across a long reasoning chain, instead of averaging it away?

HYPOTHESIS
    A PRODUCTIVE contradiction — two relators true in different contexts — is
    retained as two distinct typed relations through an arbitrarily long
    inference chain. It is never softmax-averaged into a single blended value
    (CLAUDE.md §1.1: "contradiction must be HELD, not averaged"), and its status
    never silently decays from PRODUCTIVE.

SETUP (the classic wave/particle duality)
    light EVIDENCES dual_nature   (σ=0.8)   — "light behaves as a wave"
    light NEGATES   dual_nature   (σ=0.8)   — "light does not (it's a particle)"
    NEGATES ∈ CONTRA(EVIDENCES), so these contradict on the same ordered pair.
    Both are true in different experimental contexts → marked PRODUCTIVE.
    A few non-contradictory relators provide surrounding reasoning content.

CHAIN
    Run CHAIN_LEN warp/observe steps under a goal that keeps both relations in
    scope, ticking the inference clock each step. After every step, check that
    the active field still carries BOTH parallel relators and that the log still
    holds the productive contradiction.

TRANSFORMER BASELINE
    Softmax over the two competing relations yields a single convex blend
    (≈0.5·wave + 0.5·particle): one averaged value, the tension erased. Baseline
    distinct-relations-retained = 1 (collapsed).

PASS CRITERION
    At every step t ∈ [0, CHAIN_LEN]: exactly 2 distinct typed relators persist
    on (light → dual_nature), the contradiction is held, and its status stays
    PRODUCTIVE.
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.engine.predictive import PredictiveEngine
from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator
from toris.reasoning.contradiction import ResolutionStatus

T = RelationType
CHAIN_LEN = 50
EDGE = ("light", "dual_nature")


def build():
    c = {
        n: ConceptState(id=n)
        for n in ["light", "dual_nature", "interference", "photoelectric"]
    }

    def rel(s, t, tau, sigma=0.8):
        return Relator(tau, c[s], c[t], sigma=sigma)

    field = RelationalField()
    # the productive contradiction
    r_wave = rel("light", "dual_nature", T.EVIDENCES)
    r_particle = rel("light", "dual_nature", T.NEGATES)
    field.add_relator(r_wave)
    field.add_relator(r_particle)
    # surrounding reasoning content
    field.add_relator(rel("dual_nature", "interference", T.CAUSAL))
    field.add_relator(rel("dual_nature", "photoelectric", T.EVIDENCES))

    manifold = GoalManifold(
        Goal(
            "characterize the nature of light",
            concepts={"light", "dual_nature", "interference", "photoelectric"},
            preferred_types={T.EVIDENCES, T.NEGATES, T.CAUSAL},
        )
    )
    return field, manifold, r_wave, r_particle


def run_experiment(verbose: bool = True) -> dict:
    field, manifold, r_wave, r_particle = build()
    engine = PredictiveEngine()

    # t=0: surface and mark the contradiction PRODUCTIVE.
    manifold.warp(field)
    entry = manifold.contradiction_log.mark_productive(
        r_wave, r_particle, note="wave/particle duality — both true in context"
    )

    history = []  # (t, n_parallel_distinct_types, held, productive, status)
    retained_every_step = True

    for _ in range(1, CHAIN_LEN + 1):
        t = manifold.tick()
        # warp the persistent field (active topology at this step)
        warped = manifold.warp(field, t_discovered=t)
        # simulate inference activity: observe the active structure
        engine.step(field, warped.relators())

        parallels = warped.relators_between(*EDGE)
        distinct_types = {r.tau for r in parallels}
        held = manifold.contradiction_log.held()
        productive = manifold.contradiction_log.productive()
        status = entry.resolution_status

        step_ok = (
            len(parallels) == 2
            and distinct_types == {T.EVIDENCES, T.NEGATES}
            and entry in held
            and status is ResolutionStatus.PRODUCTIVE
        )
        retained_every_step = retained_every_step and step_ok
        history.append((t, len(distinct_types), len(held), len(productive)))

    # final state on the persistent field
    final_parallels = field.relators_between(*EDGE)
    sigmas = sorted(r.sigma for r in final_parallels)
    # what a softmax baseline would have collapsed them into:
    softmax_blend = sum(sigmas) / len(sigmas)

    passed = (
        retained_every_step
        and len(final_parallels) == 2
        and entry.resolution_status is ResolutionStatus.PRODUCTIVE
    )

    if verbose:
        _print_report(entry, history, final_parallels, sigmas, softmax_blend, passed)

    return {
        "chain_len": CHAIN_LEN,
        "retained_every_step": retained_every_step,
        "final_distinct_relators": len(final_parallels),
        "final_status": entry.resolution_status.value,
        "softmax_baseline_would_keep": 1,
        "softmax_blend_value": softmax_blend,
        "passed": passed,
    }


def _print_report(entry, history, final_parallels, sigmas, softmax_blend, passed):
    print("=" * 68)
    print("TORIS Experiment 01 — Contradiction Retention")
    print("=" * 68)
    print(f"Contradiction: {entry}")
    print(f"Chain length : {CHAIN_LEN} inference steps")
    print("-" * 68)
    print("Sampled steps (t, distinct types on edge, held, productive):")
    for row in [history[0], history[len(history) // 2], history[-1]]:
        t, ntypes, held, prod = row
        print(f"  t={t:>3}:  distinct_types={ntypes}  held={held}  productive={prod}")
    print("-" * 68)
    print(
        f"TORIS retains  : {len(final_parallels)} distinct typed relations "
        f"(σ={sigmas})"
    )
    print(
        f"Softmax baseline: 1 blended value ≈ {softmax_blend:.3f} " f"(tension erased)"
    )
    print(
        f"Final status   : {entry.resolution_status.value} " f"(never auto-collapsed)"
    )
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(f"VERDICT: {verdict}  (2 distinct relations held at every step)")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
