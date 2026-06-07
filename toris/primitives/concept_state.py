"""ConceptState — the TORIS replacement for the embedding vector.

A concept is NOT a point in Euclidean space. It is a *distribution over
relational roles* (MATH_SPEC §2): Π_C : T → [0,1] on the probability simplex.
What a concept *is* depends on which relational roles it currently occupies,
and that shifts with the active goal via a Bayesian context update.

Formal object (the TORIS spec §3.2 / MATH_SPEC §2):

    C = (id, Π, Ψ, H)

where Π is the role distribution, Ψ the goal-salience function, and H the
ordered relational history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from scipy.spatial.distance import jensenshannon

from toris.primitives.relation_types import RelationType

if TYPE_CHECKING:  # avoid the Relator<->ConceptState import cycle at runtime
    from toris.primitives.relator import Relator


def _normalize(
    dist: Dict[RelationType, float],
) -> Dict[RelationType, float]:
    """Project a (partial, possibly unnormalized) role map onto the simplex.

    Missing types are treated as 0. A negative weight is clamped to 0. If the
    total mass is 0, fall back to a uniform distribution over all types so the
    simplex constraint Σ Π = 1 always holds (MATH_SPEC §2.1).
    """
    full = {t: max(0.0, float(dist.get(t, 0.0))) for t in RelationType}
    total = sum(full.values())
    if total <= 0.0:
        uniform = 1.0 / len(RelationType)
        return {t: uniform for t in RelationType}
    return {t: v / total for t, v in full.items()}


@dataclass
class ConceptState:
    """A concept as a live distribution over relational roles.

    Attributes
    ----------
    id:
        Unique concept identifier.
    role_distribution (Π):
        Probability distribution over relation types. Always normalized to the
        simplex; access via :attr:`pi`.
    goal_salience (Ψ):
        Goal-salience function Ψ : Goals → [0,1], stored as a mapping from goal
        identifier to a salience in [0,1].
    relational_history (H):
        Ordered list of past Relators that involved this concept.
    """

    id: str
    role_distribution: Dict[RelationType, float] = field(default_factory=dict)
    goal_salience: Dict[str, float] = field(default_factory=dict)
    relational_history: List["Relator"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.role_distribution = _normalize(self.role_distribution)

    # -- convenient aliases matching the mathematical notation ---------------
    @property
    def pi(self) -> Dict[RelationType, float]:
        """The role distribution Π_C (normalized)."""
        return self.role_distribution

    def role_prob(self, tau: RelationType) -> float:
        """Π_C(τ): probability that C occupies relational role ``tau``."""
        return self.role_distribution.get(tau, 0.0)

    # -- the Bayesian context update (MATH_SPEC §2.2) ------------------------
    def goal_warped_roles(
        self, goal_salience: Dict[RelationType, float]
    ) -> Dict[RelationType, float]:
        """Return Π_C^G without mutating the concept (pure).

        Bayesian update — prior role distribution × goal-relevance likelihood,
        renormalized::

            Π_C^G(τ) = Π_C(τ) · ψ_C(G, τ) / Z
            Z        = Σ_τ Π_C(τ) · ψ_C(G, τ)

        ``goal_salience`` is ψ_C(G, ·): the salience of each role τ under the
        goal G. Missing roles default to salience 0 (the goal deems them
        irrelevant). If the resulting mass is 0, falls back to uniform.
        """
        posterior = {
            t: self.role_distribution[t] * max(0.0, goal_salience.get(t, 0.0))
            for t in RelationType
        }
        return _normalize(posterior)

    def update_role_distribution(
        self, goal_salience: Dict[RelationType, float]
    ) -> Dict[RelationType, float]:
        """Apply the context update in place and return the new Π_C^G.

        The concept does not move in space — it changes what it *is*
        relationally (the TORIS spec §3.2). Use :meth:`goal_warped_roles` for a
        non-mutating preview.
        """
        self.role_distribution = self.goal_warped_roles(goal_salience)
        return self.role_distribution

    def set_role_distribution(self, dist: Dict[RelationType, float]) -> None:
        """Replace Π_C with an arbitrary (re-normalized) distribution."""
        self.role_distribution = _normalize(dist)

    # -- goal salience Ψ -----------------------------------------------------
    def salience_for(self, goal_id: str) -> float:
        """Ψ_C(G): how relevant this concept is under goal ``goal_id``."""
        return self.goal_salience.get(goal_id, 0.0)

    def set_salience(self, goal_id: str, value: float) -> None:
        """Set Ψ_C(G), clamped to [0,1]."""
        self.goal_salience[goal_id] = min(1.0, max(0.0, value))

    # -- relational history H ------------------------------------------------
    def record(self, relator: "Relator") -> None:
        """Append a Relator to this concept's ordered relational history H."""
        self.relational_history.append(relator)

    # -- role distance between contexts (MATH_SPEC §2.3) ---------------------
    def role_distance(
        self,
        goal_salience_a: Dict[RelationType, float],
        goal_salience_b: Dict[RelationType, float],
    ) -> float:
        """d_role(C, G₁, G₂) = JS(Π_C^{G₁} ‖ Π_C^{G₂}) (MATH_SPEC §2.3).

        How much this concept "changes its nature" between two contexts. Uses
        the Jensen–Shannon distance (base 2, in [0,1]) — a proper metric on the
        simplex. High distance means C plays very different roles under the two
        goals; for TORIS that is a feature, not a flaw.
        """
        order = list(RelationType)
        pi_a = self.goal_warped_roles(goal_salience_a)
        pi_b = self.goal_warped_roles(goal_salience_b)
        vec_a = [pi_a[t] for t in order]
        vec_b = [pi_b[t] for t in order]
        # jensenshannon returns the JS *distance* (a metric); nan only if a
        # vector sums to 0, which _normalize prevents.
        return float(jensenshannon(vec_a, vec_b, base=2))

    def dominant_role(self) -> Optional[RelationType]:
        """The relation type with the highest probability in Π_C, if any."""
        if not self.role_distribution:
            return None
        return max(self.role_distribution, key=self.role_distribution.get)

    def __repr__(self) -> str:
        top = self.dominant_role()
        top_name = top.name if top is not None else "none"
        return (
            f"ConceptState(id={self.id!r}, dominant_role={top_name}, "
            f"|H|={len(self.relational_history)})"
        )
