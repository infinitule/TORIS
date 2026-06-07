"""Experiment 09 — Michel Parameter Diagnostics.

Hypothesis: Michel parameters correctly diagnose field pathologies.

Four test fields:
  (a) Well-calibrated      → ρ_T ≈ 3/4,  η_T ≈ 0, ξ_T ≈ 1, δ_T ≈ 0.01
  (b) Structurally surprising → ρ_T << 3/4 (low confirmation)
  (c) Systematic type confusion → η_T != 0
  (d) Deep structural complexity → δ_T >> 0.01

Success: Michel Alert triggers for (b), (c), (d) but NOT for (a).
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.michel_parameters import compute as compute_michel, michel_alert


def make_concept(name: str) -> ConceptState:
    return ConceptState(id=name)


def make_relator(src, tgt, tau, sigma=0.8, kappa=0.6, epsilon=0.05):
    return Relator(tau=tau, src=src, tgt=tgt, sigma=sigma, kappa=kappa, epsilon=epsilon)


# -------------------------------------------------------------------------
# Field builders for each pathology

def field_well_calibrated():
    """Well-calibrated: pred ≈ obs, no type confusion, no deep structure."""
    names = ["A","B","C","D","E"]
    concepts = {n: make_concept(n) for n in names}
    pairs = [("A","B",RelationType.CAUSAL),("B","C",RelationType.ENABLES),
             ("C","D",RelationType.EVIDENCES),("D","E",RelationType.REFINES),
             ("A","E",RelationType.CONTAINS)]

    f_pred = RelationalField()
    f_obs  = RelationalField()
    for c in concepts.values():
        f_pred.add_concept(c)
        f_obs.add_concept(c)

    for s, t, tau in pairs:
        r = make_relator(concepts[s], concepts[t], tau, sigma=0.85, kappa=0.7, epsilon=0.02)
        f_pred.add_relator(r.clone(epsilon=0.0))
        f_obs.add_relator(r)

    return f_obs, f_pred, concepts


def field_structurally_surprising():
    """Structurally surprising: observed has many edges not in prediction."""
    names = ["P","Q","R","S","T","U"]
    concepts = {n: make_concept(n) for n in names}

    f_pred = RelationalField()
    f_obs  = RelationalField()
    for c in concepts.values():
        f_pred.add_concept(c)
        f_obs.add_concept(c)

    # Predicted: only one edge
    r_pred = make_relator(concepts["P"], concepts["Q"], RelationType.CAUSAL, kappa=0.7, epsilon=0.0)
    f_pred.add_relator(r_pred)

    # Observed: many NEW edges that were not predicted
    new_edges = [
        ("P","R",RelationType.ENABLES),
        ("Q","S",RelationType.CONTAINS),
        ("R","T",RelationType.ANALOGOUS),
        ("S","U",RelationType.TEMPORAL_BEFORE),
        ("T","U",RelationType.REFINES),
    ]
    # Add predicted edge back with some surprise
    f_obs.add_relator(r_pred.clone(epsilon=0.1))
    for s, t, tau in new_edges:
        r = make_relator(concepts[s], concepts[t], tau, sigma=0.6, kappa=0.5, epsilon=0.4)
        f_obs.add_relator(r)

    return f_obs, f_pred, concepts


def field_type_confusion():
    """Systematic type confusion: observed types consistently differ from predicted."""
    names = ["M","N","O","P2","Q2"]
    concepts = {n: make_concept(n) for n in names}
    pairs_pred = [
        ("M","N",RelationType.CAUSAL),
        ("N","O",RelationType.CAUSAL),
        ("O","P2",RelationType.CAUSAL),
        ("P2","Q2",RelationType.CAUSAL),
    ]
    pairs_obs = [
        ("M","N",RelationType.NEGATES),      # CAUSAL → NEGATES (contra)
        ("N","O",RelationType.VIOLATES),     # CAUSAL → VIOLATES (contra)
        ("O","P2",RelationType.CONTRADICTS), # CAUSAL → CONTRADICTS (contra)
        ("P2","Q2",RelationType.NEGATES),    # CAUSAL → NEGATES (contra)
    ]

    f_pred = RelationalField()
    f_obs  = RelationalField()
    for c in concepts.values():
        f_pred.add_concept(c)
        f_obs.add_concept(c)

    for (s,t,tau_p), (_,_,tau_o) in zip(pairs_pred, pairs_obs):
        r_p = make_relator(concepts[s], concepts[t], tau_p, kappa=0.65, epsilon=0.0)
        r_o = make_relator(concepts[s], concepts[t], tau_o, kappa=0.65, epsilon=0.6)
        f_pred.add_relator(r_p)
        f_obs.add_relator(r_o)

    return f_obs, f_pred, concepts


def field_deep_structural():
    """Deep structural complexity: many cycles → high δ_T."""
    names = [f"V{i}" for i in range(6)]
    concepts = {n: make_concept(n) for n in names}

    f_pred = RelationalField()
    f_obs  = RelationalField()
    for c in concepts.values():
        f_pred.add_concept(c)
        f_obs.add_concept(c)

    # Create a dense cyclic graph: V0→V1→V2→V3→V4→V5→V0 (big loop)
    # plus cross-edges: V0→V3, V1→V4, V2→V5 (multiple short cycles)
    edges = [
        ("V0","V1"),("V1","V2"),("V2","V3"),("V3","V4"),("V4","V5"),("V5","V0"),
        ("V0","V3"),("V1","V4"),("V2","V5"),("V3","V0"),
    ]
    for s, t in edges:
        r = make_relator(
            concepts[s], concepts[t], RelationType.CAUSAL,
            sigma=0.7, kappa=0.55, epsilon=0.5
        )
        f_pred.add_relator(r.clone(epsilon=0.0, kappa=0.5))
        f_obs.add_relator(r)

    return f_obs, f_pred, concepts


# -------------------------------------------------------------------------

def run():
    print("=" * 70)
    print("Experiment 09 — TORIS Michel Parameter Diagnostics")
    print("=" * 70)
    print(f"{'Field':<30} {'ρ_T':>6} {'η_T':>7} {'ξ_T':>7} {'δ_T':>7}  {'Alert':>6}")
    print("-" * 70)

    cases = [
        ("(a) Well-calibrated",       field_well_calibrated,     False),
        ("(b) Structurally surprising", field_structurally_surprising, True),
        ("(c) Type confusion",        field_type_confusion,      True),
        ("(d) Deep structural",       field_deep_structural,     True),
    ]

    results = {}
    for label, builder, expect_alert in cases:
        f_obs, f_pred, _ = builder()
        params = compute_michel(f_obs, f_pred)
        alert  = michel_alert(params)
        match  = alert == expect_alert
        print(
            f"{label:<30} {params.rho_T:>6.3f} {params.eta_T:>7.3f}"
            f" {params.xi_T:>7.3f} {params.delta_T:>7.4f}"
            f"  {'ALERT' if alert else 'OK':>6}  {'✓' if match else '✗'}"
        )
        results[label] = match

    print("-" * 70)
    print(f"Standard values: ρ_T=0.750  η_T=0.000  ξ_T=1.000  δ_T=0.010")

    print("\n" + "=" * 70)
    all_pass = all(results.values())
    print(f"Experiment 09 {'PASS' if all_pass else 'FAIL'} "
          f"({sum(results.values())}/{len(results)} checks)")
    print("=" * 70)
    return all_pass, results


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)
