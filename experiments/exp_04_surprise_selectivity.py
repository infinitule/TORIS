"""Experiment 04 — Surprise Selectivity.

Failure mode (CLAUDE.md §1.4 / §7.4): does computation concentrate on anomaly,
not noise?

HYPOTHESIS
    In a field of 20 predicted relators where only 3 carry genuine surprise,
    the propagation gate (ε > θ_ε, MATH_SPEC §3.3) routes computation to those
    3 alone. Confirmed predictions are suppressed at source and consume no
    compute. Target bar (CLAUDE.md §7.4): > 70% of processing events fall on
    the surprising relators.

MINIMAL TEST FIELD
    F_pred : 20 CAUSAL relators on edges e0..e19, σ = 0.8.
    F_obs  : 17 confirmed (same type; two with sub-threshold strength noise),
             plus 3 surprises —
               • e17  CAUSAL → NEGATES   (type surprise, D_type = 1.0)
               • e18  CAUSAL → VIOLATES  (type surprise, D_type = 0.7)
               • e19  predicted edge replaced by a brand-new edge (structural)

TRANSFORMER BASELINE
    A transformer attends over all 20 relations every forward pass: softmax
    weights are dense, so all 20 receive non-trivial gradient/compute. There is
    no mechanism to spend ~zero compute on a confirmed prediction. Expected
    baseline concentration ≈ 3/20 = 15% (uniform-ish), NOT > 70%.

PASS CRITERION
    fraction of processing events on the 3 intended surprises > 0.70.
"""

from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from toris.constants import THETA_EPSILON
from toris.engine.predictive import PredictiveEngine
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator

N_RELATORS = 20
SURPRISE_BAR = 0.70


def build_fields():
    """Return (F_pred, F_obs, surprising_edges)."""
    # 21 concepts feed 20 directed edges e_i = (n_{2i} → n_{2i+1})... instead we
    # use a fresh src/tgt pair per edge for clarity of the (src,tgt) identity.
    concepts = {i: ConceptState(id=f"c{i}") for i in range(N_RELATORS + 2)}

    def src(i):
        return concepts[i]

    def tgt(i):
        return concepts[i + 1]

    # --- F_pred: 20 CAUSAL relators, σ = 0.8 ---
    pred = RelationalField()
    predicted = {}
    for i in range(N_RELATORS):
        r = Relator(RelationType.CAUSAL, src(i), tgt(i), sigma=0.8)
        predicted[i] = r
        pred.add_relator(r)

    # --- F_obs ---
    obs = RelationalField()
    surprising_edges = set()

    # 17 confirmed (e0..e16). Two of them carry tiny strength noise that stays
    # below θ_ε to show strength-only deviation does NOT trigger propagation.
    for i in range(17):
        sigma = 0.8
        if i in (0, 1):
            sigma = 0.75  # Δσ = 0.05 → ε ≈ 0.1·0.0025 ≈ 0.00025, suppressed
        obs.add_relator(Relator(RelationType.CAUSAL, src(i), tgt(i), sigma=sigma))

    # Surprise 1: type flip to NEGATES on e17 (D_type = 1.0).
    obs.add_relator(Relator(RelationType.NEGATES, src(17), tgt(17), sigma=0.8))
    surprising_edges.add((f"c17", f"c18"))

    # Surprise 2: type flip to VIOLATES on e18 (D_type = 0.7, unrelated).
    obs.add_relator(Relator(RelationType.VIOLATES, src(18), tgt(18), sigma=0.8))
    surprising_edges.add((f"c18", f"c19"))

    # Surprise 3: structural — e19 not observed; a brand-new edge appears.
    obs.add_relator(
        Relator(RelationType.CAUSAL, concepts[N_RELATORS], concepts[N_RELATORS + 1])
    )
    surprising_edges.add((f"c{N_RELATORS}", f"c{N_RELATORS + 1}"))

    return pred, obs, surprising_edges


def run_experiment(verbose: bool = True) -> dict:
    pred, obs, surprising_edges = build_fields()

    engine = PredictiveEngine()
    report = engine.compute_delta(pred, obs)
    propagating = engine.propagate(report)

    propagating_edges = {r.edge for r in propagating}
    n_events = len(propagating)
    on_surprise = sum(1 for r in propagating if r.edge in surprising_edges)
    concentration = on_surprise / n_events if n_events else 0.0
    fraction_of_field = n_events / obs.num_relators()

    passed = concentration > SURPRISE_BAR and propagating_edges == surprising_edges

    if verbose:
        _print_report(
            report,
            obs,
            surprising_edges,
            propagating,
            concentration,
            fraction_of_field,
            passed,
        )

    return {
        "delta_s": report.delta_s,
        "n_observed": obs.num_relators(),
        "n_processing_events": n_events,
        "concentration_on_surprise": concentration,
        "fraction_of_field_processed": fraction_of_field,
        "propagating_edges": propagating_edges,
        "surprising_edges": surprising_edges,
        "passed": passed,
    }


def _print_report(
    report, obs, surprising_edges, propagating, concentration, fraction, passed
):
    print("=" * 68)
    print("TORIS Experiment 04 — Surprise Selectivity")
    print("=" * 68)
    print(f"θ_ε (propagation threshold)      : {THETA_EPSILON}")
    print(f"Observed relators                : {obs.num_relators()}")
    print(f"Intended surprises               : {len(surprising_edges)}")
    print(f"Aggregate ΔS                     : {report.delta_s:.4f}")
    print(
        f"  ΔS_struct={report.delta_s_struct:.4f}  "
        f"ΔS_type={report.delta_s_type:.4f}  "
        f"ΔS_strength={report.delta_s_strength:.4f}"
    )
    print("-" * 68)
    print("Per-relator surprise (ε) — sorted, propagating marked ▶")
    for rs in sorted(
        report.per_relator.values(), key=lambda x: x.epsilon, reverse=True
    ):
        mark = "▶" if rs.propagates() else " "
        print(
            f"  {mark} {rs.relator.edge[0]:>3}→{rs.relator.edge[1]:<3} "
            f"{rs.relator.tau.name:<12} ε={rs.epsilon:.4f}"
            f"  (struct={rs.eps_struct:.0f} type={rs.eps_type:.2f} "
            f"str={rs.eps_strength:.4f})"
        )
    print("-" * 68)
    print(f"Processing events (propagating)  : {len(propagating)}")
    print(f"Fraction of field processed      : {fraction:.1%}")
    print(f"Compute concentration on surprise: {concentration:.1%}")
    print(f"Transformer baseline (dense)     : ~{3/obs.num_relators():.1%}")
    print("-" * 68)
    verdict = "PASS ✅" if passed else "FAIL ❌"
    print(f"VERDICT: {verdict}  (bar: > {SURPRISE_BAR:.0%} on the surprises)")
    print("=" * 68)


if __name__ == "__main__":
    import sys

    result = run_experiment()
    sys.exit(0 if result["passed"] else 1)
