"""Layer 3 — the Goal Manifold and the goal-driven warp operator Φ."""

from toris.goal.manifold import Goal, GoalManifold
from toris.goal.subgoal import Subgoal, SubgoalStatus
from toris.goal.warp import (
    concept_overlap,
    goal_relevance_multiplier,
    relevance,
    type_fit,
    warp_field,
)

__all__ = [
    "Goal",
    "GoalManifold",
    "Subgoal",
    "SubgoalStatus",
    "relevance",
    "concept_overlap",
    "type_fit",
    "goal_relevance_multiplier",
    "warp_field",
]
