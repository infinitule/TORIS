"""Subgoals — the active stack of a Goal Manifold (MATH_SPEC §4.1).

A subgoal is the unit that modulates the warp operator's salience recomputation
(MATH_SPEC §4.2 step 1). Per §4.1 each subgoal carries:

    g = (description, priority ∈ [0,1], blocking, parent_goal)

The relevance function (§4.3) additionally consults the goal's *concept set* and
*goal type*, so a subgoal also carries ``concepts`` (the concept ids it bears
on) and ``preferred_types`` (the relation types useful for achieving it). The
primary goal ``Goal`` (in ``manifold.py``) carries the same two fields, so the
relevance function treats G_p and any subgoal uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Set

from toris.primitives.relation_types import RelationType


class SubgoalStatus(Enum):
    """Which list of the manifold a subgoal currently lives in."""

    ACTIVE = "ACTIVE"  # in S_active
    RESOLVED = "RESOLVED"  # in S_resolved (retained for backtracking)
    ABANDONED = "ABANDONED"  # in S_abandoned (negative space — paths rejected)


@dataclass
class Subgoal:
    """An active goal-stack element that warps the field (MATH_SPEC §4.1)."""

    description: str
    priority: float = 1.0  # ∈ [0,1]
    blocking: bool = False
    parent_goal: Optional[str] = None
    concepts: Set[str] = field(default_factory=set)  # concept ids in scope
    preferred_types: Set[RelationType] = field(default_factory=set)
    status: SubgoalStatus = SubgoalStatus.ACTIVE

    def __post_init__(self) -> None:
        self.priority = min(1.0, max(0.0, float(self.priority)))

    def __repr__(self) -> str:
        return (
            f"Subgoal({self.description!r}, priority={self.priority:.2f}, "
            f"blocking={self.blocking}, {self.status.value})"
        )
