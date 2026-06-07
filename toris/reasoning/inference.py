"""The full TORIS inference loop — all layers tied together.

One step of inference runs the predictive-coding cycle and then rewrites the
field via fast plasticity (MATH_SPEC §3, §5.1):

    1. tick the goal manifold's clock
    2. project   F_pred = Φ(G, F)              (goal-warped expectation)
    3. observe   F_obs  = incoming structure
    4. surprise  ε(R), ΔS  (compute_delta + propagate)
    5. rewrite   F^{t+1} = Φ(G, F ⊕ ΔF^t)      (fast plasticity)

The loop snapshots the field's topology at construction so structural drift
(§5.3) can be measured after a reasoning chain. It uses local surprise as the
only signal — there is no global loss function (the TORIS spec §1.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from toris.engine.predictive import Observable, PredictiveEngine
from toris.field.relational_field import RelationalField
from toris.goal.warp import relevance_fn, warp_field
from toris.plasticity.fast import (
    FastPlasticity,
    TopologySnapshot,
    snapshot,
    structural_drift,
)


@dataclass
class StepRecord:
    """A per-step trace of the inference loop."""

    t: int
    delta_s: float
    n_added: int
    n_strengthened: int
    n_weakened: int
    n_suppressed: int
    n_relators: int


class InferenceLoop:
    """Drives predictive coding + fast plasticity over a relational field."""

    def __init__(
        self,
        field: RelationalField,
        manifold,
        engine: Optional[PredictiveEngine] = None,
        plasticity: Optional[FastPlasticity] = None,
    ) -> None:
        self.field = field
        self.manifold = manifold
        self.engine = engine or PredictiveEngine()
        self.plasticity = plasticity or FastPlasticity()
        self.initial_snapshot: TopologySnapshot = snapshot(field)  # F^0
        self.history: List[StepRecord] = []

    def step(self, incoming: Observable) -> StepRecord:
        """Run one full inference step and return its trace."""
        self.manifold.tick()

        # Use warp_field (not just project) so that contradiction scanning
        # fires at each step and the ContradictionLog stays current (§4.2 step 4)
        f_pred = warp_field(self.manifold, self.field)
        f_obs = self.engine.observe(incoming)
        report = self.engine.compute_delta(f_pred, f_obs)
        self.engine.propagate(report)

        new_field, delta = self.plasticity.step(self.field, report, self.manifold)
        self.field = new_field

        record = StepRecord(
            t=self.manifold.t,
            delta_s=report.delta_s,
            n_added=len(delta.added),
            n_strengthened=len(delta.strengthened),
            n_weakened=len(delta.weakened),
            n_suppressed=len(delta.suppressed),
            n_relators=self.field.num_relators(),
        )
        self.history.append(record)
        return record

    def run(self, observations: Iterable[Observable]) -> List[StepRecord]:
        """Run a sequence of observations as a reasoning chain."""
        return [self.step(obs) for obs in observations]

    def drift(self) -> dict:
        """Structural drift d_topo(F^0, F^T) since construction (MATH_SPEC §5.3)."""
        return structural_drift(self.initial_snapshot, snapshot(self.field))
