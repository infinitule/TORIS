"""The Goal Manifold — live inference-time goal structure (MATH_SPEC §4.1).

    G = (G_p, S_active, S_resolved, S_abandoned, L_contra)

The manifold is not a static query. It warps the relational field (Φ, §4.2),
holds the contradiction log L_contra, and tracks the negative space of
abandoned subgoals. The primary goal G_p is invariant during a reasoning chain;
subgoals move between the active stack, the resolved set, and the abandoned set
(the TORIS spec §3.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set

from toris.constants import THETA_AMPLIFY, THETA_KAPPA
from toris.field.relational_field import RelationalField
from toris.goal import warp as _warp
from toris.goal.subgoal import Subgoal, SubgoalStatus
from toris.primitives.relation_types import RelationType
from toris.reasoning.contradiction import ContradictionLog


@dataclass
class Goal:
    """The primary goal G_p — invariant during a reasoning chain.

    Carries the concept set and preferred relation types the relevance function
    consults (MATH_SPEC §4.3). An empty ``concepts`` / ``preferred_types`` means
    the goal does not constrain on that axis (neutral relevance).
    """

    description: str
    concepts: Set[str] = field(default_factory=set)
    preferred_types: Set[RelationType] = field(default_factory=set)

    def __repr__(self) -> str:
        return f"Goal({self.description!r}, |concepts|={len(self.concepts)})"


@dataclass
class GoalManifold:
    """A live goal structure that warps the field and holds contradictions."""

    primary: Goal
    active: List[Subgoal] = field(default_factory=list)
    resolved: List[Subgoal] = field(default_factory=list)
    abandoned: List[Subgoal] = field(default_factory=list)
    contradiction_log: ContradictionLog = field(default_factory=ContradictionLog)
    t: int = 0  # inference clock (step index for contradiction timestamps)

    # -- subgoal stack management ------------------------------------------
    def add_subgoal(self, subgoal: Subgoal) -> Subgoal:
        """Push a subgoal onto the active stack S_active."""
        subgoal.status = SubgoalStatus.ACTIVE
        self.active.append(subgoal)
        return subgoal

    def resolve_subgoal(self, subgoal: Subgoal) -> None:
        """Move a subgoal from active to resolved (retained for backtracking)."""
        self._move(subgoal, self.resolved, SubgoalStatus.RESOLVED)

    def abandon_subgoal(self, subgoal: Subgoal) -> None:
        """Move a subgoal from active to abandoned (negative space)."""
        self._move(subgoal, self.abandoned, SubgoalStatus.ABANDONED)

    def _move(
        self, subgoal: Subgoal, dest: List[Subgoal], status: SubgoalStatus
    ) -> None:
        if subgoal in self.active:
            self.active.remove(subgoal)
        subgoal.status = status
        if subgoal not in dest:
            dest.append(subgoal)

    # -- the warp operator Φ -----------------------------------------------
    def warp(
        self,
        field: RelationalField,
        theta_kappa: float = THETA_KAPPA,
        theta_amplify: float = THETA_AMPLIFY,
        t_discovered: Optional[int] = None,
    ) -> RelationalField:
        """Φ(self, field) → F': warp the field and surface contradictions (§4.2)."""
        return _warp.warp_field(self, field, theta_kappa, theta_amplify, t_discovered)

    def relevance(self, relator) -> float:
        """The goal-relevance multiplier for a relator (§4.2 step 1)."""
        return _warp.goal_relevance_multiplier(self, relator)

    def tick(self) -> int:
        """Advance the inference clock by one step and return the new value."""
        self.t += 1
        return self.t

    def __repr__(self) -> str:
        return (
            f"GoalManifold(primary={self.primary.description!r}, "
            f"active={len(self.active)}, resolved={len(self.resolved)}, "
            f"abandoned={len(self.abandoned)}, "
            f"contradictions={len(self.contradiction_log)}, t={self.t})"
        )
