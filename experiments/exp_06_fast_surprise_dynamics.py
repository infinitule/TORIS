import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import time
import random
from typing import List, Tuple

from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.field.relational_field import RelationalField
from toris.engine.surprise import SurpriseMetric
from toris.engine.wave import CyclicWaveEngine
from toris.constants import THETA_EPSILON

def test_fast_surprise_dynamics():
    print("Starting Exp 06: Fast Surprise Dynamics Verification\n")

    # 1. Setup Field
    field = RelationalField()
    concepts = [ConceptState(id=f"C{i}") for i in range(20)]

    # Background noise: 90 random relators (low salience)
    for i in range(90):
        src, tgt = random.sample(concepts, 2)
        field.add_relator(Relator(
            tau=RelationType.CAUSAL,
            src=src,
            tgt=tgt,
            sigma=0.5,
            kappa=0.05,
            epsilon=0.05
        ))

    # Ground Truth: 3 genuinely surprising relators
    surprising_relators = []
    for i in range(3):
        src, tgt = random.sample(concepts, 2)
        r = Relator(
            tau=RelationType.CAUSAL,
            src=src,
            tgt=tgt,
            sigma=0.8,
            kappa=0.9,
            epsilon=0.9
        )
        field.add_relator(r)
        surprising_relators.append(r)

    # Cycle 1: Stable
    c18, c19 = concepts[18], concepts[19]
    field.add_relator(Relator(RelationType.CAUSAL, c18, c19, kappa=0.1, epsilon=0.1))
    field.add_relator(Relator(RelationType.CAUSAL, c19, c18, kappa=0.1, epsilon=0.1))

    # Cycle 2: Unstable (high salience drive)
    c0, c1, c2 = concepts[0], concepts[1], concepts[2]
    field.add_relator(Relator(RelationType.CAUSAL, c0, c1, kappa=0.9, epsilon=0.5))
    field.add_relator(Relator(RelationType.CAUSAL, c1, c2, kappa=0.9, epsilon=0.5))
    field.add_relator(Relator(RelationType.CAUSAL, c2, c0, kappa=0.9, epsilon=0.5))

    # 2. Define Prediction
    f_pred = RelationalField()
    for r in field.relators():
        if r not in surprising_relators and not (r.src_id in ["C0", "C1", "C2"] and r.tgt_id in ["C0", "C1", "C2"]):
            f_pred.add_relator(r.clone())

    metric = SurpriseMetric()
    # Use b=0.1 to ensure the unstable cycle exceeds the threshold (sin(0.9)/0.1 = 7.8 > 5.0)
    wave_engine = CyclicWaveEngine(b=0.1)

    # 3. Compare Pipelines
    print("Running Full ΔS Pipeline...")
    start_full = time.time()
    report_full = metric.report(f_pred, field)
    end_full = time.time()
    time_full = end_full - start_full

    print("Running TFSA Pipeline...")
    start_tfsa = time.time()
    report_tfsa = metric.combined_pipeline(f_pred, field)
    end_tfsa = time.time()
    time_tfsa = end_tfsa - start_tfsa

    # 4. Verifications
    print("\n--- Results ---")
    prop_full = set(id(r) for r in report_full.propagating())
    prop_tfsa = set(id(r) for r in report_tfsa.propagating())

    intersection = prop_full & prop_tfsa
    union = prop_full | prop_tfsa
    overlap = len(intersection) / len(union) if union else 1.0
    print(f"Pipeline overlap: {overlap:.2%} (Qualitative equivalence)")

    print(f"Full ΔS time: {time_full:.6f}s")
    print(f"TFSA time:   {time_tfsa:.6f}s")
    print(f"Speedup:     {time_full/time_tfsa:.2f}x")

    unstable = wave_engine.scan_field(field)
    print(f"Unstable cycles found: {len(unstable)}")
    for res in unstable:
        print(f"  - {res}")

    has_unstable_cycle = any("C0" in res.concepts and "C1" in res.concepts and "C2" in res.concepts for res in unstable)
    print(f"C0-C1-C2 unstable: {has_unstable_cycle}")

    diff = len(prop_full ^ prop_tfsa)
    print(f"Propagating set difference: {diff} relators")

    if overlap > 0.9 and has_unstable_cycle:
        print("\nSUCCESS: Exp 06 passed.")
    else:
        print("\nFAILURE: Exp 06 failed.")

if __name__ == "__main__":
    test_fast_surprise_dynamics()
