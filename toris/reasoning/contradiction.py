"""The Contradiction Log — held tension as a computational resource.

TORIS never resolves contradiction by averaging (no softmax over contradicting
relators — the TORIS spec §1.1). Instead, contradictions are HELD as live pairs in a
log (the TORIS spec §3.5 / MATH_SPEC §4.1):

    L_contra entry = (R_a, R_b, t_discovered, resolution_status)

    resolution_status ∈ {LIVE, RESOLVED, DEFERRED, PRODUCTIVE}

A ``PRODUCTIVE`` contradiction is one where the tension *itself* is the answer —
both relators are true in different contexts. Productive contradictions are
NEVER collapsed. This module owns step 4 of the warp operator Φ (MATH_SPEC §4.2):
surfacing the goal-relevant contradictions found in the active topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Iterator, List, Optional, Union

from toris.primitives.relator import Relator


class ResolutionStatus(Enum):
    """The lifecycle status of a logged contradiction (the TORIS spec §3.5)."""

    LIVE = "LIVE"  # unresolved tension, currently held
    RESOLVED = "RESOLVED"  # one side won; tension discharged
    DEFERRED = "DEFERRED"  # set aside, to revisit later
    PRODUCTIVE = "PRODUCTIVE"  # both true in different contexts — NEVER collapsed

    def is_held(self) -> bool:
        """True iff this status keeps both relators alive (not collapsed)."""
        return self is not ResolutionStatus.RESOLVED


ContradictionKey = FrozenSet[int]


@dataclass
class ContradictionEntry:
    """A single held contradiction: two relators and how it stands."""

    relator_a: Relator
    relator_b: Relator
    t_discovered: int
    resolution_status: ResolutionStatus = ResolutionStatus.LIVE
    note: str = ""

    @property
    def key(self) -> ContradictionKey:
        """Identity of the contradiction: the unordered pair of relator ids."""
        return frozenset({self.relator_a.rid, self.relator_b.rid})

    @property
    def is_productive(self) -> bool:
        return self.resolution_status is ResolutionStatus.PRODUCTIVE

    @property
    def is_held(self) -> bool:
        """True while the tension is retained (LIVE, DEFERRED, or PRODUCTIVE)."""
        return self.resolution_status.is_held()

    def __repr__(self) -> str:
        return (
            f"Contradiction({self.relator_a.tau.name} ⊗ "
            f"{self.relator_b.tau.name} on "
            f"{self.relator_a.src_id}→{self.relator_a.tgt_id}, "
            f"t={self.t_discovered}, {self.resolution_status.value})"
        )


EntryRef = Union[ContradictionEntry, ContradictionKey, Relator]


class ContradictionLog:
    """A live registry of held contradictions (L_contra).

    Entries are keyed by the unordered pair of relator ids, so the same
    contradiction is never logged twice and re-detecting it across inference
    steps does not disturb its status — a PRODUCTIVE contradiction stays
    productive however many times the field is re-warped.
    """

    def __init__(self) -> None:
        self._entries: Dict[ContradictionKey, ContradictionEntry] = {}

    # -- detection ----------------------------------------------------------
    @staticmethod
    def detect_contradiction(r_a: Relator, r_b: Relator) -> bool:
        """R_a ⊗ R_b: do these two relators contradict? (MATH_SPEC §1.3)."""
        return r_a.contradicts(r_b)

    # -- logging ------------------------------------------------------------
    def log_contradiction(
        self,
        r_a: Relator,
        r_b: Relator,
        t_discovered: int = 0,
        status: ResolutionStatus = ResolutionStatus.LIVE,
        note: str = "",
    ) -> ContradictionEntry:
        """Log a contradiction between two relators (idempotent on the pair).

        Requires R_a ⊗ R_b (raises ``ValueError`` otherwise — we never log a
        non-contradiction). If this pair is already logged, the existing entry
        is returned unchanged, preserving any PRODUCTIVE/DEFERRED status.
        """
        if not self.detect_contradiction(r_a, r_b):
            raise ValueError(
                f"relators do not contradict: {r_a.tau.name} vs {r_b.tau.name} "
                f"on {r_a.edge} / {r_b.edge}"
            )
        key = frozenset({r_a.rid, r_b.rid})
        existing = self._entries.get(key)
        if existing is not None:
            return existing
        entry = ContradictionEntry(
            relator_a=r_a,
            relator_b=r_b,
            t_discovered=t_discovered,
            resolution_status=status,
            note=note,
        )
        self._entries[key] = entry
        return entry

    def scan_field(self, field, t_discovered: int = 0) -> List[ContradictionEntry]:
        """Surface every contradiction among parallel relators in ``field``.

        Implements MATH_SPEC §4.2 step 4 over an already-warped (active) field:
        contradictions can only arise between relators on the same ordered pair,
        so we examine the parallel relators on each edge. Returns the entries
        (new or pre-existing) touched by this scan.
        """
        touched: List[ContradictionEntry] = []
        for src, tgt in field.edge_set():
            parallels = field.relators_between(src, tgt)
            for i in range(len(parallels)):
                for j in range(i + 1, len(parallels)):
                    if self.detect_contradiction(parallels[i], parallels[j]):
                        touched.append(
                            self.log_contradiction(
                                parallels[i], parallels[j], t_discovered
                            )
                        )
        return touched

    # -- status transitions -------------------------------------------------
    def _resolve_key(
        self, ref: EntryRef, other: Optional[Relator] = None
    ) -> ContradictionKey:
        if isinstance(ref, ContradictionEntry):
            return ref.key
        if isinstance(ref, Relator) and isinstance(other, Relator):
            return frozenset({ref.rid, other.rid})
        if isinstance(ref, frozenset):
            return ref
        raise TypeError("provide an entry, a key, or two relators")

    def mark_productive(
        self, ref: EntryRef, other: Optional[Relator] = None, note: str = ""
    ) -> ContradictionEntry:
        """Mark a contradiction PRODUCTIVE — both sides true in different contexts.

        Productive contradictions are never collapsed; this is the status that
        encodes "the tension itself is the answer" (the TORIS spec §3.5).
        """
        return self._set_status(ref, ResolutionStatus.PRODUCTIVE, other, note)

    def mark_resolved(
        self, ref: EntryRef, other: Optional[Relator] = None, note: str = ""
    ) -> ContradictionEntry:
        """Mark a contradiction RESOLVED — the tension has been discharged."""
        return self._set_status(ref, ResolutionStatus.RESOLVED, other, note)

    def mark_deferred(
        self, ref: EntryRef, other: Optional[Relator] = None, note: str = ""
    ) -> ContradictionEntry:
        """Mark a contradiction DEFERRED — set aside to revisit."""
        return self._set_status(ref, ResolutionStatus.DEFERRED, other, note)

    def _set_status(
        self,
        ref: EntryRef,
        status: ResolutionStatus,
        other: Optional[Relator],
        note: str,
    ) -> ContradictionEntry:
        key = self._resolve_key(ref, other)
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError("contradiction not in log")
        entry.resolution_status = status
        if note:
            entry.note = note
        return entry

    # -- queries ------------------------------------------------------------
    def get(self, r_a: Relator, r_b: Relator) -> Optional[ContradictionEntry]:
        return self._entries.get(frozenset({r_a.rid, r_b.rid}))

    def entries(self) -> List[ContradictionEntry]:
        return list(self._entries.values())

    def live(self) -> List[ContradictionEntry]:
        return [
            e
            for e in self._entries.values()
            if e.resolution_status is ResolutionStatus.LIVE
        ]

    def productive(self) -> List[ContradictionEntry]:
        return [e for e in self._entries.values() if e.is_productive]

    def held(self) -> List[ContradictionEntry]:
        """Every contradiction still retained (i.e. not RESOLVED)."""
        return [e for e in self._entries.values() if e.is_held]

    def __len__(self) -> int:
        return len(self._entries)

    def log_from_wave(
        self,
        unstable_cycles: List[Any],
        field,
        t_discovered: int = 0,
    ) -> List[ContradictionEntry]:
        """Log structural instabilities as implicit contradictions (§9.7).

        These are dynamic tensions that emerge from field oscillations and may not
        satisfy the static contradiction check. We bypass the check and log the
        most surprising pair in the unstable cycle.
        """
        entries: List[ContradictionEntry] = []
        for res in unstable_cycles:
            if len(res.relators) < 2:
                continue

            # Pick the two relators with highest final surprise as the tension poles
            sorted_relators = sorted(res.relators, key=lambda r: r.epsilon, reverse=True)
            r_a, r_b = sorted_relators[0], sorted_relators[1]

            key = frozenset({r_a.rid, r_b.rid})
            entry = ContradictionEntry(
                relator_a=r_a,
                relator_b=r_b,
                t_discovered=t_discovered,
                resolution_status=ResolutionStatus.LIVE,
                note=f"wave-detected instability in cycle {res.concepts}",
            )
            self._entries[key] = entry
            entries.append(entry)
        return entries
