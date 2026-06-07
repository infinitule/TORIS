"""Slow Plasticity — the training analog (the TORIS spec §2; three-timescale §8).

The slowest timescale. Where medium plasticity consolidates within a handful of
sessions (§5.2), slow plasticity forms the *long-term baseline* across many
sessions and protects consolidated structure from fast decay. It is the closest
TORIS analog to "training": durable knowledge that persists once a relator has
proven repeatedly useful.

MATH_SPEC specifies only fast (§5.1) and medium (§5.2). The slow timescale is
named in the directory blueprint ("Slow plasticity (training analog)") but its
equations are unspecified, so this is a documented extension (DEVIATION D-17):

    baseline^{n+1}(R) = baseline^n(R) + η_slow · [σ^n(R) − baseline^n(R)]

A relator whose baseline rises above a consolidation threshold becomes
*consolidated* (long-term knowledge); ``apply_floor`` then protects it from
dropping below that baseline, so fast WEAKEN cannot erase durable structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set

from toris.constants import CONSOLIDATION_THRESHOLD, ETA_SLOW
from toris.field.relational_field import RelationalField


@dataclass
class SlowRecord:
    """One slow-consolidation update for a relator's long-term baseline."""

    rid: int
    baseline_before: float
    baseline_after: float
    consolidated: bool


class SlowPlasticity:
    """Maintains a protected long-term baseline σ per relator (training analog)."""

    def __init__(
        self,
        eta_slow: float = ETA_SLOW,
        consolidation_threshold: float = CONSOLIDATION_THRESHOLD,
    ) -> None:
        self.eta_slow = eta_slow
        self.consolidation_threshold = consolidation_threshold
        self.baseline: Dict[int, float] = {}
        self.consolidated: Set[int] = set()

    def consolidate(self, field: RelationalField) -> List[SlowRecord]:
        """EMA each relator's current σ into its long-term baseline.

        A relator whose baseline reaches ``consolidation_threshold`` is marked
        consolidated (durable knowledge). Returns the per-relator records.
        """
        records: List[SlowRecord] = []
        for r in field.relators():
            before = self.baseline.get(r.rid, r.sigma)
            after = before + self.eta_slow * (r.sigma - before)
            self.baseline[r.rid] = after
            now_consolidated = after >= self.consolidation_threshold
            if now_consolidated:
                self.consolidated.add(r.rid)
            records.append(SlowRecord(r.rid, before, after, r.rid in self.consolidated))
        return records

    def apply_floor(self, field: RelationalField) -> None:
        """Protect consolidated relators: σ cannot fall below its baseline.

        This is how durable knowledge resists fast-plasticity decay — the slow
        baseline is a floor for any relator that has been consolidated.
        """
        for r in field.relators():
            if r.rid in self.consolidated:
                r.sigma = max(r.sigma, self.baseline[r.rid])

    def baseline_of(self, rid: int) -> float:
        return self.baseline.get(rid, 0.0)

    def is_consolidated(self, rid: int) -> bool:
        return rid in self.consolidated

    def __repr__(self) -> str:
        return (
            f"SlowPlasticity(η_slow={self.eta_slow}, "
            f"consolidated={len(self.consolidated)})"
        )
