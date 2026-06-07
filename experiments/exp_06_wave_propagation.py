import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.engine.fsd_pipeline import FastSurpriseDynamics, FSDReport

def run_exp_06():
    print("--- Experiment 06: Wave Propagation Verification ---")

    # 1. Setup: 6-relator directed loop
    # Concepts: C0 -> C1 -> C2 -> C3 -> C4 -> C5 -> C0
    concepts = [ConceptState(id=f"C{i}") for i in range(6)]

    # Create relators R1...R6
    relators = []
    for i in range(6):
        r = Relator(
            tau=RelationType.CAUSAL,
            src=concepts[i],
            tgt=concepts[(i + 1) % 6],
            sigma=1.0,
            kappa=0.5, # Moderate salience for all
            epsilon=0.0
        )
        relators.append(r)

    # R1 is surprising
    relators[0].epsilon = 0.8

    # Mock RelationalField
    class MockField:
        def __init__(self, relators):
            self.relators = relators
            self.graph = nx.MultiDiGraph()
            for r in relators:
                self.graph.add_edge(r.src_id, r.tgt_id, **{"relator": r})

    field = MockField(relators)

    # 2. Discrete Propagation Baseline (conceptual)
    # In discrete propagation, only direct neighbors of surprising relators activate.
    # R1 (surprising) -> activates R2.
    # R2 is not surprising, so it doesn't activate R3.
    discrete_activated_count = 1 # Only R2
    print(f"Discrete propagation activated count: {discrete_activated_count}")

    # 3. FSD Wave Propagation
    fsd = FastSurpriseDynamics(theta_epsilon=0.2)
    # F_pred is not used in this simple mock, so we pass None
    report = fsd.compute(field, None, None)

    fsd_activated = report.activated_relators
    fsd_count = len(fsd_activated)
    print(f"FSD wave propagation activated count: {fsd_count}")

    # 4. Verification
    print(f"Activation Ratio (FSD/Discrete): {fsd_count / discrete_activated_count:.2f}x")

    assert fsd_count >= 2 * discrete_activated_count, \
        f"FSD failed to activate significantly more relators: {fsd_count} vs {discrete_activated_count}"

    print("SUCCESS: FSD activates >= 2x more relators in the loop.")

    # 5. Print Wave Trajectory (conceptual)
    # Since our integrate function returns the steady state, we'll re-run
    # a manual integration to capture the trajectory.
    from toris.engine.fsd_wave import CyclicSurpriseSystem, RK4_STEP_H

    system = CyclicSurpriseSystem(relators, b=0.2)
    init_eps = np.array([r.epsilon for r in relators])

    trajectory = []
    curr_eps = init_eps
    t_wave = 5 / 0.2
    steps = int(t_wave / RK4_STEP_H)

    for t in range(steps):
        trajectory.append(curr_eps.copy())
        curr_eps = system.rk4_step(curr_eps)

    trajectory = np.array(trajectory)

    plt.figure(figsize=(10, 6))
    for i in range(6):
        plt.plot(trajectory[:, i], label=f"R{i+1}")

    plt.title("Surprise Wave Trajectory in 6-Relator Loop")
    plt.xlabel("Step")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.grid(True)
    plt.savefig("exp_06_trajectory.png")
    print("Trajectory plot saved as exp_06_trajectory.png")

if __name__ == "__main__":
    run_exp_06()
