"""Dedicated surprise-propagation module (the TORIS spec §2, MATH_SPEC §3.3–3.4).

Separates the *propagation* concern from prediction/observation in
predictive.py.  The core idea:

    Only Relators with ε > θ_ε propagate their surprise signal.
    Confirmed predictions (ε ≤ θ_ε) are suppressed — they consume no compute.

This module provides:
  - `PropagationGraph`  — tracks which relators forwarded surprise and to whom
  - `propagate_surprise` — standalone function matching MATH_SPEC §3.3 rule
  - `multi_hop_propagation` — cascade surprise K hops through the field
  - `SurpriseFront` — the live wavefront of propagating surprise at step t

The PredictiveEngine.propagate() method is kept as the single call-site for the
inference loop; this module supplies the heavy-lifting primitives it delegates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from toris.constants import THETA_EPSILON
from toris.engine.surprise import SurpriseReport, RelatorSurprise
from toris.field.relational_field import RelationalField
from toris.primitives.relator import Relator


@dataclass
class PropagationEvent:
    """One hop in the propagation graph."""
    relator: Relator
    epsilon: float
    hop: int                  # 0 = initial surprise, 1 = first cascade, …
    source_id: Optional[str]  # concept that forwarded surprise into this relator


@dataclass
class SurpriseFront:
    """The active wavefront of surprise at a given step.

    Attributes:
        events:   All propagation events in this step (sorted by hop).
        concepts: Unique concept ids that participated in propagation.
        total_surprise: Sum of ε over all propagating relators.
    """
    events: List[PropagationEvent] = field(default_factory=list)
    concepts: Set[str] = field(default_factory=set)
    total_surprise: float = 0.0

    def add(self, event: PropagationEvent) -> None:
        self.events.append(event)
        self.concepts.add(event.relator.src_id)
        self.concepts.add(event.relator.tgt_id)
        self.total_surprise += event.epsilon

    def propagating_relators(self) -> List[Relator]:
        return [e.relator for e in self.events]

    def __len__(self) -> int:
        return len(self.events)


def propagate_surprise(
    report: SurpriseReport,
    theta: float = THETA_EPSILON,
) -> SurpriseFront:
    """Apply the propagation gate from MATH_SPEC §3.3.

    For each relator in *report*:
      - record ε onto the relator's .epsilon attribute
      - gate: only ε > θ_ε produces a PropagationEvent at hop 0

    Returns a SurpriseFront containing all relators that passed the gate.
    """
    front = SurpriseFront()
    for rs in report.per_relator.values():
        rs.relator.epsilon = rs.epsilon  # materialise measured surprise
        if rs.epsilon > theta:
            front.add(PropagationEvent(
                relator=rs.relator,
                epsilon=rs.epsilon,
                hop=0,
                source_id=None,
            ))
    return front


def multi_hop_propagation(
    initial_front: SurpriseFront,
    field: RelationalField,
    max_hops: int = 3,
    decay: float = 0.5,
    theta: float = THETA_EPSILON,
) -> SurpriseFront:
    """Cascade surprise K hops through the relational field.

    Starting from *initial_front*, follow outgoing edges of each surprising
    concept and forward (decay · ε) to the next relator.  Gate again at θ_ε.

    The cascade stops when:
      (a) max_hops is reached, or
      (b) the front becomes empty (all surprise below θ_ε)

    This implements the sparse propagation property from MATH_SPEC §3.4:
    confirmed predictions are never visited.

    Args:
        initial_front: SurpriseFront from propagate_surprise() at hop 0.
        field:         The live RelationalField to traverse.
        max_hops:      Maximum cascade depth.  Default 3.
        decay:         Multiplicative surprise decay per hop ∈ (0,1).
        theta:         Propagation threshold.

    Returns:
        A combined SurpriseFront with events at hops 0..max_hops.
    """
    combined = SurpriseFront()
    for e in initial_front.events:
        combined.add(e)

    current_concepts: Set[str] = set(initial_front.concepts)
    visited: Set[str] = set()  # relator ids already propagated

    for hop in range(1, max_hops + 1):
        if not current_concepts:
            break
        next_concepts: Set[str] = set()
        for r in field.relators():
            rid = id(r)
            if rid in visited:
                continue
            if r.src_id not in current_concepts:
                continue
            # Forward decayed surprise
            propagated_eps = decay * r.epsilon
            if propagated_eps > theta:
                event = PropagationEvent(
                    relator=r,
                    epsilon=propagated_eps,
                    hop=hop,
                    source_id=r.src_id,
                )
                combined.add(event)
                next_concepts.add(r.tgt_id)
                visited.add(rid)
        current_concepts = next_concepts

    return combined


class PropagationGraph:
    """Accumulates the full propagation history across multiple steps.

    Useful for experiment 04 (surprise selectivity) and structural drift
    analysis — shows *which* relators consistently receive surprise, and which
    are perpetually silent (candidates for weakening).
    """

    def __init__(self) -> None:
        self._hit_counts: Dict[int, int] = {}   # id(relator) → hit count
        self._total_eps: Dict[int, float] = {}  # id(relator) → cumulative ε

    def record(self, front: SurpriseFront) -> None:
        for event in front.events:
            rid = id(event.relator)
            self._hit_counts[rid] = self._hit_counts.get(rid, 0) + 1
            self._total_eps[rid] = self._total_eps.get(rid, 0.0) + event.epsilon

    def hot_relator_ids(self, top_n: int = 10) -> List[int]:
        """Return ids of the top_n most-hit relators (by hit count)."""
        return sorted(
            self._hit_counts, key=lambda k: self._hit_counts[k], reverse=True
        )[:top_n]

    def average_epsilon(self, relator_id: int) -> float:
        hits = self._hit_counts.get(relator_id, 0)
        if hits == 0:
            return 0.0
        return self._total_eps.get(relator_id, 0.0) / hits

    def selectivity_ratio(self, n_total_relators: int, top_n: int) -> float:
        """Fraction of surprise concentrated in top_n relators.

        Returns [0,1]; near 1.0 = highly selective (§7 criterion 4 target > 0.7).
        """
        if not self._total_eps:
            return 0.0
        top_ids = self.hot_relator_ids(top_n)
        top_eps = sum(self._total_eps.get(i, 0.0) for i in top_ids)
        total_eps = sum(self._total_eps.values())
        return top_eps / total_eps if total_eps else 0.0
