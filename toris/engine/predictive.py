"""The Predictive Engine — TORIS's inference cycle (MATH_SPEC §3.1, §3.3).

The engine runs predictive coding over the relational field:

    project  →  observe  →  compute_delta  →  propagate

* **project**       F_pred = project(F, G): what we expect the field to be.
* **observe**       F_obs: the relational structure that actually arrived.
* **compute_delta** per-relator ε(R) and aggregate ΔS via the surprise metric.
* **propagate**     only relators with ε > θ_ε propagate; confirmed predictions
                    are suppressed at source and consume no compute.

This module is *inference-time*: it uses local surprise as the primary signal
and never a global loss function (the TORIS spec §1.1).
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Tuple, Union

from toris.engine.surprise import SurpriseMetric, SurpriseReport
from toris.field.relational_field import RelationalField
from toris.primitives.relator import Relator

# A goal-relevance multiplier for a relator (supplied by the Goal Manifold).
RelevanceFn = Callable[[Relator], float]
Observable = Union[RelationalField, Iterable[Relator]]


class PredictiveEngine:
    """Runs the project/observe/delta/propagate cycle over a relational field."""

    def __init__(self, metric: Optional[SurpriseMetric] = None) -> None:
        self.metric = metric or SurpriseMetric()

    # -- project (MATH_SPEC §3.1) ------------------------------------------
    def project(
        self,
        field: RelationalField,
        relevance: Optional[RelevanceFn] = None,
    ) -> RelationalField:
        """F_pred = project(F, G): the field we expect to observe next.

        The reference projection is a *continuity prior*: we predict the current
        relational structure persists. When a goal-relevance function is given,
        the prediction is the goal-warped field Φ(G, F) (MATH_SPEC §4.2) — the
        goal changes which relations we expect to matter. Returns a fresh field
        (relators cloned, rids preserved) so prediction and observation stay
        distinct objects.
        """
        if relevance is not None:
            return field.warp(relevance)
        return field.copy()

    # -- observe (MATH_SPEC §3.1) ------------------------------------------
    def observe(self, incoming: Observable) -> RelationalField:
        """F_obs: wrap the incoming relational structure as a field."""
        if isinstance(incoming, RelationalField):
            return incoming
        obs = RelationalField()
        obs.add_relators(incoming)
        return obs

    # -- compute_delta (MATH_SPEC §3.2–3.3) --------------------------------
    def compute_delta(
        self, f_pred: RelationalField, f_obs: RelationalField
    ) -> SurpriseReport:
        """Per-relator ε(R) and aggregate ΔS between prediction and observation."""
        return self.metric.report(f_pred, f_obs)

    # -- propagate (MATH_SPEC §3.3) ----------------------------------------
    def propagate(self, report: SurpriseReport) -> List[Relator]:
        """Apply the propagation gate; return the relators that propagate.

        Records each observed relator's measured surprise onto its ε field, then
        gates: only ε > θ_ε propagates. Confirmed predictions (ε ≤ θ_ε) are
        suppressed — they are not returned and trigger no downstream compute.
        """
        for rs in report.per_relator.values():
            rs.relator.epsilon = rs.epsilon  # record measured surprise
        return report.propagating()

    # -- full cycle convenience --------------------------------------------
    def step(
        self,
        field: RelationalField,
        incoming: Observable,
        relevance: Optional[RelevanceFn] = None,
    ) -> Tuple[SurpriseReport, List[Relator]]:
        """Run one full project→observe→delta→propagate cycle.

        Returns the surprise report and the list of propagating relators (the
        active set on which downstream computation concentrates).
        """
        f_pred = self.project(field, relevance)
        f_obs = self.observe(incoming)
        report = self.compute_delta(f_pred, f_obs)
        propagating = self.propagate(report)
        return report, propagating
