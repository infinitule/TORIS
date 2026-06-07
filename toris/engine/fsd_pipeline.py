import time
import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Dict, List, Set, Any
from toris.engine.fsd_fast_approx import tfsa, newton_refine
from toris.engine.fsd_wave import CyclicSurpriseSystem

# Numerical constants from Section 9.4
ALPHA_BLEND: float = 0.7
DEFAULT_THETA_EPSILON: float = 0.2

@dataclass
class FSDReport:
    epsilon_final: Dict[int, float]
    activated_relators: List[Any]
    loop_count: int
    wave_contributions: Dict[int, float]
    timing: Dict[str, float]

class FastSurpriseDynamics:
    """
    The unified FSD Pipeline.
    Combines TFSA (local surprise) and Cyclic Surprise Waves (propagated surprise).
    """
    def __init__(self, theta_epsilon: float = DEFAULT_THETA_EPSILON):
        self.theta_epsilon = theta_epsilon

    def compute(self, field, goal_manifold, f_pred) -> FSDReport:
        """
        Computes the final surprise epsilon for all relators in the observed field.

        Args:
            field: The RelationalField object.
            goal_manifold: The GoalManifold object.
            f_pred: The predicted field.

        Returns:
            FSDReport containing the surprise analysis.
        """
        start_time = time.perf_counter()

        # 1. TFSA: Local Surprise Approximation
        # We iterate over all relators in the current field
        eps_refined = {}
        for r in field.relators: # Assuming field.relators is an iterable of Relators
            e_fast = tfsa(r.kappa)
            eps_refined[r.rid] = newton_refine(e_fast, r.kappa)

        t_tfsa = time.perf_counter()

        # 2. Loop Detection and Wave Propagation
        # We need the graph structure to find cycles.
        # Assuming field.graph is the networkx.MultiDiGraph
        loops = list(nx.simple_cycles(field.graph))
        loop_count = len(loops)

        # Map relators to their wave surprise.
        # Default is eps_refined (no wave contribution)
        eps_wave = {r.rid: eps_refined[r.rid] for r in field.relators}

        for loop_nodes in loops:
            # a loop is a sequence of nodes: [n1, n2, ..., nk, n1]
            # We need to identify the relators (edges) between them.
            loop_relators = []
            for i in range(len(loop_nodes)):
                u = loop_nodes[i]
                v = loop_nodes[(i + 1) % len(loop_nodes)]
                # Find the relator between u and v.
                # In a MultiDiGraph, there might be multiple; we take the most salient one.
                edges = field.graph.get_edge_data(u, v)
                if edges:
                    # In MultiDiGraph, edges is a dict of {key: attr_dict}
                    # We find the relator in the attr_dict with max kappa
                    best_r = None
                    max_k = -1.0
                    for attr_dict in edges.values():
                        r = attr_dict.get("relator")
                        if r and r.kappa > max_k:
                            max_k = r.kappa
                            best_r = r
                    if best_r:
                        loop_relators.append(best_r)

            if len(loop_relators) < 2:
                continue

            # Create surprise system for this loop
            system = CyclicSurpriseSystem(loop_relators, b=self.theta_epsilon)

            # Initial epsilon for the loop is the refined local surprise
            init_eps = np.array([eps_refined[r.rid] for r in loop_relators])

            # Integrate to find steady-state wave surprise
            final_eps_vec = system.integrate(init_eps)

            # Update eps_wave for relators in this loop
            for r, val in zip(loop_relators, final_eps_vec):
                eps_wave[r.rid] = val

        t_wave = time.perf_counter()

        # 3. Blend and Gate
        epsilon_final = {}
        activated_relators = []

        for r in field.relators:
            # Blend: 70% local, 30% wave
            val = ALPHA_BLEND * eps_refined[r.rid] + (1.0 - ALPHA_BLEND) * eps_wave[r.rid]
            epsilon_final[r.rid] = val

            if val > self.theta_epsilon:
                activated_relators.append(r)

        total_time = time.perf_counter() - start_time

        return FSDReport(
            epsilon_final=epsilon_final,
            activated_relators=activated_relators,
            loop_count=loop_count,
            wave_contributions=eps_wave,
            timing={
                "tfsa": t_tfsa - start_time,
                "wave": t_wave - t_tfsa,
                "total": total_time
            }
        )
