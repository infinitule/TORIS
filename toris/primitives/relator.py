"""The Relator — TORIS's base computational primitive (replaces the neuron).

A Relator is a typed, directional, surprise-bearing transformation between two
ConceptStates (the TORIS spec §3.1 / MATH_SPEC §1.1):

    R = (τ, A, B, σ, κ, ε)

    τ ∈ T      relation type
    A          source ConceptState (domain)
    B          target ConceptState (codomain)
    σ ∈ [0,1]  strength (confidence in the relation)
    κ ∈ [0,1]  contextual salience (activation under the current goal)
    ε ∈ ℝ≥0    surprise (deviation from prediction)

Relators are *asymmetric by default*: R(A→B, CAUSAL) does not imply
R(B→A, CAUSAL). This module also implements the two algebra operators over
relators: composition ∘_τ (MATH_SPEC §1.2) and contradiction ⊗ (§1.3).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import (
    RelationType,
    composition_rule,
    contra,
    is_symmetric,
)

# Process-local monotonic counter for stable, readable relator identities.
_rid_counter = itertools.count(1)


def _next_rid() -> int:
    return next(_rid_counter)


@dataclass
class Relator:
    """A typed, directional relation carrying strength, salience, and surprise.

    σ and κ are clamped to [0,1] and ε to [0, ∞) on construction and should be
    kept in range by mutators (plasticity caps σ at 1.0). ``rid`` is an
    auto-assigned identity so a relator stays trackable as its σ/κ/ε mutate.
    """

    tau: RelationType
    src: ConceptState
    tgt: ConceptState
    sigma: float = 1.0  # σ — strength / confidence
    kappa: float = 1.0  # κ — contextual salience
    epsilon: float = 0.0  # ε — surprise
    rid: int = field(default_factory=_next_rid)

    def __post_init__(self) -> None:
        self.sigma = _clamp01(self.sigma)
        self.kappa = _clamp01(self.kappa)
        self.epsilon = max(0.0, float(self.epsilon))

    # -- identity of the underlying directed edge ---------------------------
    @property
    def edge(self) -> tuple[str, str]:
        """The directed edge (src_id, tgt_id) this relator realizes."""
        return (self.src.id, self.tgt.id)

    @property
    def src_id(self) -> str:
        return self.src.id

    @property
    def tgt_id(self) -> str:
        return self.tgt.id

    # -- symmetry (the TORIS spec §3.1) ------------------------------------------
    @property
    def is_symmetric(self) -> bool:
        """True iff this relator's type may be reversed without changing meaning."""
        return is_symmetric(self.tau)

    def reverse(self) -> "Relator":
        """Return the inverse relator B→A, only if the type is symmetric.

        Asymmetric relators have no meaning-preserving inverse, so reversing one
        raises ``ValueError`` (MATH_SPEC: symmetry must be explicitly declared).
        The reversed relator carries the same σ, κ, ε but a fresh identity.
        """
        if not self.is_symmetric:
            raise ValueError(
                f"cannot reverse asymmetric relator of type {self.tau.name}; "
                "only symmetric types may be reversed"
            )
        return Relator(
            tau=self.tau,
            src=self.tgt,
            tgt=self.src,
            sigma=self.sigma,
            kappa=self.kappa,
            epsilon=self.epsilon,
        )

    # -- contradiction operator ⊗ (MATH_SPEC §1.3) --------------------------
    def contradicts(self, other: "Relator") -> bool:
        """R ⊗ other: do these two relators structurally contradict?

        True iff they connect the same ordered pair (same src, same tgt) and
        this relator's type lies in CONTRA(other.type)::

            R₁ ⊗ R₂ ⇔ src₁=src₂ ∧ tgt₁=tgt₂ ∧ τ₁ ∈ CONTRA(τ₂)

        An explicit ``CONTRADICTS`` relator over the same pair is also treated
        as a contradiction with any non-identical relation on that pair, since
        it names the contradiction directly (MATH_SPEC §1.3).
        """
        if (self.src_id, self.tgt_id) != (other.src_id, other.tgt_id):
            return False
        if self.tau in contra(other.tau) or other.tau in contra(self.tau):
            return True
        # Explicit naming: a CONTRADICTS relator opposes any differing relation
        # asserted over the same ordered pair.
        if RelationType.CONTRADICTS in (self.tau, other.tau):
            return self.tau != other.tau
        return False

    # -- composition operator ∘_τ (MATH_SPEC §1.2) --------------------------
    def can_compose(self, other: "Relator") -> bool:
        """True iff ``self ∘ other`` is defined: tgt(self)=src(other) under Ω."""
        if self.tgt_id != other.src_id:
            return False
        return composition_rule(self.tau, other.tau) is not None

    def compose(self, other: "Relator") -> Optional["Relator"]:
        """Compose two relators along a shared concept (MATH_SPEC §1.2).

        Defined iff ``tgt(self) == src(other)`` and (τ₁, τ₂) ∈ Ω. Returns::

            R₁ ∘ R₂ = (τ_composed, src(R₁), tgt(R₂),
                       σ_rule(σ₁,σ₂), min(κ₁,κ₂), ε₁+ε₂)

        Surprise accumulates *additively* along the chain — long chains are
        inherently more uncertain. Returns ``None`` when composition is
        undefined (including any CONTRADICTS first operand, which blocks Ω).
        """
        if self.tgt_id != other.src_id:
            return None
        rule = composition_rule(self.tau, other.tau)
        if rule is None:
            return None
        return Relator(
            tau=rule.result_type,
            src=self.src,
            tgt=other.tgt,
            sigma=_clamp01(rule.strength_rule(self.sigma, other.sigma)),
            kappa=min(self.kappa, other.kappa),
            epsilon=self.epsilon + other.epsilon,
        )

    # -- copying ------------------------------------------------------------
    def clone(self, **overrides: object) -> "Relator":
        """Return a copy of this relator, preserving ``rid`` (identity).

        Endpoints (ConceptStates) are shared by reference — cloning copies the
        *relation*, not the concepts. Any of ``tau, src, tgt, sigma, kappa,
        epsilon`` may be overridden. Used by field copy/warp so a relator stays
        the "same" edge as its σ/κ/ε mutate across predicted/observed fields.
        """
        params = {
            "tau": self.tau,
            "src": self.src,
            "tgt": self.tgt,
            "sigma": self.sigma,
            "kappa": self.kappa,
            "epsilon": self.epsilon,
            "rid": self.rid,
        }
        params.update(overrides)
        return Relator(**params)  # type: ignore[arg-type]

    # -- history bookkeeping ------------------------------------------------
    def register(self) -> "Relator":
        """Record this relator in the relational history H of both endpoints.

        Kept explicit (not done at construction) so building a relator has no
        side effects; call this when the relator becomes part of a field.
        """
        self.src.record(self)
        if self.tgt is not self.src:
            self.tgt.record(self)
        return self

    def __repr__(self) -> str:
        return (
            f"Relator(#{self.rid} {self.src_id} —{self.tau.name}→ {self.tgt_id}, "
            f"σ={self.sigma:.3f}, κ={self.kappa:.3f}, ε={self.epsilon:.3f})"
        )


def _clamp01(x: float) -> float:
    """Clamp a value to the closed unit interval [0,1]."""
    return min(1.0, max(0.0, float(x)))
