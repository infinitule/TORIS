"""Experiment 02 — Goal-Warp Sensitivity.

Failure mode (CLAUDE.md §1.4 / §7.2): does changing the goal change *which
relations are even active* — not just attention scores, but the topology of the
field?

HYPOTHESIS
    One fixed relational field, warped under two different goal manifolds,
    yields two different active topologies (MATH_SPEC §4.2). Different edges
    survive suppression, and different contradictions surface to the
    contradiction log (step 4). A concept shared by both goals (smoke) even
    changes its relational role-distribution between the two contexts
    (d_role > 0, MATH_SPEC §2.3).

ONE FIELD, TWO DOMAINS
    Physical : oxygen/fuel ENABLES fire ; fire CAUSAL smoke ; and a held
               contradiction on (oxygen → fire): CAUSAL vs NEGATES.
    Procedure: smoke CAUSAL alarm ; alarm {CONDITIONAL, TEMPORAL_BEFORE}
               evacuation ; and a held contradiction on (alarm → evacuation):
               CONDITIONAL vs CONTRADICTS.

TWO GOALS
    G_phys "explain the fire"     : concepts {fire,oxygen,fuel,smoke},
                                    types {CAUSAL,ENABLES,NEGATES}
    G_proc "plan the evacuation"  : concepts {alarm,evacuation,drill,smoke},
                                    types {CONDITIONAL,TEMPORAL_BEFORE,CONTRADICTS}

TRANSFORMER BASELINE
    Attention reweights *all* tokens for every query; the graph of which
    relations exist is fixed. Changing the prompt/goal changes soft weights but
    never removes an edge or surfaces a structurally-different contradiction.
    Expected baseline: identical active edge set under both goals.

PASS CRITERION
    active_edges(G_phys) ≠ active_edges(G_proc) AND the surfaced contradiction
    edges differ between the two goals.
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator

T = RelationType


def build_field():
    c = {
        n: ConceptState(id=n)
        for n in [
            "fire",
            "oxygen",
            "fuel",
            "smoke",
            "alarm",
            "evacuation",
            "drill",
        ]
    }

    def rel(s, t, tau, sigma=0.9):
        return Relator(tau, c[s], c[t], sigma=sigma)

    f = RelationalField()
    # physical domain
    f.add_relator(rel("oxygen", "fire", T.ENABLES))
    f.add_relator(rel("fuel", "fire", T.ENABLES))
    f.add_relator(rel("fire", "smoke", T.CAUSAL))
    # held contradiction on (oxygen → fire)
    f.add_relator(rel("oxygen", "fire", T.CAUSAL))
    f.add_relator(rel("oxygen", "fire", T.NEGATES))
    # bridge + procedure domain
    f.add_relator(rel("smoke", "alarm", T.CAUSAL))
    f.add_relator(rel("alarm", "evacuation", T.CONDITIONAL))
    f.add_relator(rel("alarm", "evacuation", T.TEMPORAL_BEFORE))
    # held contradiction on (alarm → evacuation)
    f.add_relator(rel("alarm", "evacuation", T.CONTRADICTS))
    return f, c


def _salience(preferred):
    return {t: 1.0 for t in preferred}


def run_experiment(verbose: bool = True) -> dict:
    field, concepts = build_field()

    g_phys = GoalManifold(
        Goal(
            "explain the fire",
            concepts={"fire", "oxygen", "fuel", "smoke"},
            preferred_types={T.CAUSAL, T.ENABLES, T.NEGATES},
        )
    )
    g_proc = GoalManifold(
        Goal(
            "plan the evacuation",
            concepts={"alarm", "evacuation", "drill", "smoke"},
            preferred_types={T.CONDITIONAL, T.TEMPORAL_BEFORE, T.CONTRADICTS},
        )
    )

    warped_phys = g_phys.warp(field)
    warped_proc = g_proc.warp(field)

    edges_phys = warped_phys.edge_set()
    edges_proc = warped_proc.edge_set()

    contra_phys = {(e.relator_a.edge) for e in g_phys.contradiction_log.entries()}
    contra_proc = {(e.relator_a.edge) for e in g_proc.contradiction_log.entries()}

    # shared concept "smoke" changes its relational role-distribution (§2.3)
    smoke = concepts["smoke"]
    d_role = smoke.role_distance(
        _salience(g_phys.primary.preferred_types),
        _salience(g_proc.primary.preferred_types),
    )

    edges_differ = edges_phys != edges_proc
    contra_differ = contra_phys != contra_proc
    passed = edges_differ and contra_differ

    if verbose:
        _print_report(
            edges_phys,
            edges_proc,
            contra_phys,
            contra_proc,
            d_role,
            passed,
        )

    return {
        "edges_phys": edges_phys,
        "edges_proc": edges_proc,
        "only_phys": edges_phys - edges_proc,
        "only_proc": edges_proc - edges_phys,
        "contra_phys": contra_phys,
        "contra_proc": contra_proc,
        "d_role_smoke": d_role,
        "edges_differ": edges_differ,
        "contra_differ": contra_differ,
        "passed": passed,
    }


def _print_report(edges_phys, edges_proc, contra_phys, contra_proc, d_role, passed):
    print("=" * 68)
    print("TORIS Experiment 02 — Goal-Warp Sensitivity")
    print("=" * 68)
    print(f"Active edges under G_phys ({len(edges_phys)}): {sorted(edges_phys)}")
    print(f"Active edges under G_proc ({len(edges_proc)}): {sorted(edges_proc)}")
    print("-" * 68)
    print(f"Active ONLY under G_phys : {sorted(edges_phys - edges_proc)}")
    print(f"Active ONLY under G_proc : {sorted(edges_proc - edges_phys)}")
    print(f"Active under both        : {sorted(edges_phys & edges_proc)}")
    print("-" * 68)
    print(f"Contradictions surfaced — G_phys: {sorted(contra_phys)}")
    print(f"Contradictions surfaced — G_proc: {sorted(contra_proc)}")
    print("-" * 68)
    print(f"Role-distance of shared concept 'smoke' between goals: {d_role:.4f}")
    print("Transformer baseline: identical active edge set under both goals.")
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(f"VERDICT: {verdict}  (edges differ AND contradictions differ)")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
