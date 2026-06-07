"""The Cyclic Surprise Wave (Section 9.4).

Surprise does not only propagate one discrete hop at a time; through relational
*cycles* it flows as a damped wave. For a cycle of N relators each relator's
surprise is driven by the sine of the next relator's salience:

    dε_i/dt = sin(κ_{(i+1) mod N}) − b · ε_i          (b = damping = θ_ε)

Discretized with Euler step h. The fixed point is ε_i* = sin(κ_{i+1})/b, so
high-salience relators downstream drive surprise upstream — you are surprised by
what you expected to matter but that behaved unexpectedly. A cycle whose
sustained surprise climbs above the instability threshold is flagged as an
*implicit* contradiction: a structural tension that never announced itself with
a CONTRADICTS relator but emerges from the field's own dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx

from toris.constants import (
    H_EULER,
    INSTABILITY_THRESHOLD,
    N_WAVE_STEPS,
    THETA_EPSILON,
)
from toris.field.relational_field import RelationalField
from toris.primitives.relator import Relator


# ---------------------------------------------------------------------------
# The discrete wave (Section 9.4, implementable form)
# ---------------------------------------------------------------------------

def cyclic_wave_step(
    epsilons: List[float],
    kappas: List[float],
    b: float,
    h: float = H_EULER,
) -> List[float]:
    """One Euler step of the cyclic surprise wave equation (§9.4).

    ``ε_i ← ε_i + h·(sin(κ_{(i+1) mod N}) − b·ε_i)``.
    """
    n = len(epsilons)
    out: List[float] = []
    for i in range(n):
        drive = math.sin(kappas[(i + 1) % n])
        decay = b * epsilons[i]
        out.append(epsilons[i] + h * (drive - decay))
    return out


def run_wave(
    epsilons: List[float],
    kappas: List[float],
    b: float,
    n_steps: int = N_WAVE_STEPS,
    h: float = H_EULER,
) -> List[List[float]]:
    """Run the cyclic wave for ``n_steps`` Euler steps; return the full history.

    The history is a list of successive ε-vectors, ``history[0]`` being the
    initial state and ``history[-1]`` the final state.
    """
    history = [list(epsilons)]
    current = list(epsilons)
    for _ in range(n_steps):
        current = cyclic_wave_step(current, kappas, b, h)
        history.append(current)
    return history


def detect_instability(
    history: List[List[float]],
    threshold: float = INSTABILITY_THRESHOLD,
) -> bool:
    """True iff sustained surprise exceeds the instability threshold (§9.4).

    A cycle that settles below the threshold has resolved (stable); one whose
    final surprise stays above it carries genuine, sustained tension (unstable).
    The default threshold is 1/θ_ε = 5.0 (§9.9).
    """
    if not history:
        return False
    return max(history[-1]) > threshold


# ---------------------------------------------------------------------------
# Field-level cycle analysis
# ---------------------------------------------------------------------------

@dataclass
class CycleResult:
    """The outcome of running the wave on one relational cycle."""

    concepts: List[str]  # ordered concept ids forming the cycle
    relators: List[Relator]  # one representative relator per hop
    kappas: List[float]
    final_epsilons: List[float]
    is_unstable: bool

    @property
    def length(self) -> int:
        return len(self.relators)

    def __repr__(self) -> str:
        loop = "→".join(self.concepts + [self.concepts[0]]) if self.concepts else ""
        state = "UNSTABLE" if self.is_unstable else "stable"
        return f"CycleResult({loop}, N={self.length}, {state})"


class CyclicWaveEngine:
    """Finds relational cycles and classifies them via the surprise wave (§9.4)."""

    def __init__(
        self,
        b: float = THETA_EPSILON,
        n_steps: int = N_WAVE_STEPS,
        h: float = H_EULER,
        instability_threshold: float = INSTABILITY_THRESHOLD,
    ) -> None:
        self.b = b
        self.n_steps = n_steps
        self.h = h
        self.instability_threshold = instability_threshold

    # -- cycle discovery ----------------------------------------------------
    @staticmethod
    def _digraph(field: RelationalField) -> nx.DiGraph:
        """A simple directed graph view of the field (parallels collapsed)."""
        g = nx.DiGraph()
        for r in field.relators():
            g.add_edge(r.src_id, r.tgt_id)
        return g

    def find_cycles(self, field: RelationalField) -> List[List[str]]:
        """All simple directed cycles (length ≥ 2) in the field, via networkx."""
        cycles = [c for c in nx.simple_cycles(self._digraph(field)) if len(c) >= 2]
        return cycles

    def _cycle_relators(
        self, cycle: List[str], field: RelationalField
    ) -> Optional[List[Relator]]:
        """Strongest representative relator for each hop around the cycle."""
        relators: List[Relator] = []
        n = len(cycle)
        for i in range(n):
            parallels = field.relators_between(cycle[i], cycle[(i + 1) % n])
            if not parallels:
                return None  # cycle node order had no direct edge (shouldn't happen)
            relators.append(max(parallels, key=lambda r: r.sigma))
        return relators

    # -- per-cycle analysis -------------------------------------------------
    def analyze_cycle(
        self,
        cycle: List[str],
        field: RelationalField,
        b: Optional[float] = None,
    ) -> Optional[CycleResult]:
        """Run the wave on a cycle and classify it stable/unstable."""
        relators = self._cycle_relators(cycle, field)
        if relators is None:
            return None
        b = self.b if b is None else b
        kappas = [r.kappa for r in relators]
        epsilons = [r.epsilon for r in relators]  # seed from current surprise
        history = run_wave(epsilons, kappas, b, self.n_steps, self.h)
        unstable = detect_instability(history, self.instability_threshold)
        return CycleResult(
            concepts=list(cycle),
            relators=relators,
            kappas=kappas,
            final_epsilons=history[-1],
            is_unstable=unstable,
        )

    def scan_field(
        self, field: RelationalField, b: Optional[float] = None
    ) -> List[CycleResult]:
        """Analyze every cycle; return only the unstable ones (§9.5 step 4)."""
        results: List[CycleResult] = []
        for cycle in self.find_cycles(field):
            res = self.analyze_cycle(cycle, field, b)
            if res is not None and res.is_unstable:
                results.append(res)
        return results
