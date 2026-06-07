"""Medium Plasticity — cross-session consolidation (MATH_SPEC §5.2).

Where fast plasticity rewrites topology *within* an inference chain (§5.1),
medium plasticity moves each relator's baseline strength *across* sessions toward
the surprise level it sustained:

    σ^{s+1}(R) = σ^s(R) + η_med · [ε_accumulated(R, session) − σ^s(R)]

This is a moving average toward the session's surprise level. Relators that are
consistently surprising (repeatedly needed across sessions) gradually increase
their baseline strength — they consolidate. Relators that are never surprising
gradually fade toward zero. The "surprise level" is a magnitude in [0,1], so
``ε_accumulated`` is taken as the session-mean per-relator surprise (see D-16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from toris.constants import ETA_MED
from toris.engine.surprise import SurpriseReport
from toris.field.relational_field import RelationalField
from toris.primitives.relator import Relator


@dataclass
class ConsolidationRecord:
    """What one §5.2 consolidation did to a relator's baseline strength."""

    rid: int
    sigma_before: float
    sigma_after: float
    surprise_level: float  # ε_accumulated(R, session) ∈ [0,1]


class MediumPlasticity:
    """Accumulates session surprise and consolidates σ across sessions (§5.2)."""

    def __init__(self, eta_med: float = ETA_MED) -> None:
        self.eta_med = eta_med
        # per-relator running surprise for the *current* session
        self._sum: Dict[int, float] = {}
        self._count: Dict[int, int] = {}
        self.session_history: List[List[ConsolidationRecord]] = []

    # -- accumulate surprise during a session ------------------------------
    def observe(self, relator: Relator, epsilon: float) -> None:
        """Record one surprise sample for ``relator`` in the current session."""
        self._sum[relator.rid] = self._sum.get(relator.rid, 0.0) + epsilon
        self._count[relator.rid] = self._count.get(relator.rid, 0) + 1

    def observe_report(self, report: SurpriseReport) -> None:
        """Accumulate every relator's surprise from one inference step."""
        for rs in report.per_relator.values():
            self.observe(rs.relator, rs.epsilon)

    def surprise_level(self, rid: int) -> float:
        """ε_accumulated(R, session): the session-mean surprise, clamped to [0,1].

        A relator not observed this session has level 0 (it will fade).
        """
        count = self._count.get(rid, 0)
        if count == 0:
            return 0.0
        return min(1.0, max(0.0, self._sum[rid] / count))

    # -- consolidation at session boundary (the §5.2 update) ---------------
    def consolidate(self, field: RelationalField) -> List[ConsolidationRecord]:
        """Apply σ^{s+1} = σ^s + η_med·(ε_acc − σ^s) to every relator, then reset.

        Mutates the relators' σ in place and returns the per-relator records.
        The session accumulators are cleared so the next session starts fresh.
        """
        records: List[ConsolidationRecord] = []
        for r in field.relators():
            target = self.surprise_level(r.rid)
            before = r.sigma
            r.sigma = min(1.0, max(0.0, before + self.eta_med * (target - before)))
            records.append(ConsolidationRecord(r.rid, before, r.sigma, target))
        self._reset_session()
        self.session_history.append(records)
        return records

    def _reset_session(self) -> None:
        self._sum.clear()
        self._count.clear()

    def __repr__(self) -> str:
        return (
            f"MediumPlasticity(η_med={self.eta_med}, "
            f"sessions={len(self.session_history)})"
        )
