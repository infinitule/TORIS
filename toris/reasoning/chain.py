"""Reasoning Chain executor — sparse generalization across K hops (MATH_SPEC §6).

TORIS reaches a K-hop conclusion from N seed relators (N < K) by walking an
expected concept route, reusing seed relators where they exist and instantiating
*hypothetical* connectors (σ = 0.1) where they are missing, then composing the
chain and reporting a calibrated uncertainty.

The conclusion strength obeys the §6 lower bound:

    σ_conclusion ≥ (min_σ)^K · exp(−λ · Σ_i ΔS_i)

A broken chain (a hop whose endpoints do not exist) yields σ = 0. Otherwise the
conclusion degrades *gracefully* with chain length and accumulated surprise — it
never collapses to zero, and the system always knows how uncertain it is, because
the uncertainty is structurally embedded in the chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from toris.constants import ALPHA, LAMBDA, SIGMA_HYPOTHETICAL
from toris.field.relational_field import RelationalField
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import RelationType
from toris.primitives.relator import Relator


@dataclass
class ChainResult:
    """The outcome of a sparse K-hop inference."""

    path: List[Relator]
    hops: int  # K
    n_seed: int  # relators found in the field
    n_hypothetical: int  # connectors instantiated to bridge gaps
    min_sigma: float  # minimum strength along the chain
    sum_surprise: float  # Σ_i ΔS_i (additive over hops)
    sigma_chain: float  # realized Π σ_i (the composed strength)
    sigma_bound: float  # §6 lower bound (min_σ)^K · exp(−λ·ΣΔS)
    broken: bool  # a hop could not be realized at all
    composite: Optional[Relator]  # the folded R₁∘…∘R_K, if types compose

    def __repr__(self) -> str:
        return (
            f"ChainResult(K={self.hops}, seeds={self.n_seed}, "
            f"hypotheticals={self.n_hypothetical}, min_σ={self.min_sigma:.3f}, "
            f"σ_chain={self.sigma_chain:.3e}, σ_bound={self.sigma_bound:.3e}, "
            f"broken={self.broken})"
        )


class ReasoningChain:
    """Executes relator chains: composition, hypotheticals, and §6 uncertainty."""

    def __init__(self, lam: float = LAMBDA) -> None:
        self.lam = lam  # surprise-decay constant λ

    # -- hypothetical instantiation (MATH_SPEC §6) -------------------------
    @staticmethod
    def instantiate_hypothetical(
        src: ConceptState,
        tgt: ConceptState,
        tau: RelationType = RelationType.CAUSAL,
    ) -> Relator:
        """A bridging connector for a missing hop: σ = 0.1, high surprise.

        The act of positing a connector the field never predicted is itself a
        structural surprise, so ε is the structural-surprise weight α (ε_struct =
        1 ⇒ ε = α, MATH_SPEC §3.3). This makes hypothetical hops dominate ΣΔS,
        which is exactly why heavily-hypothesized conclusions are reported as
        very uncertain.
        """
        return Relator(tau, src, tgt, sigma=SIGMA_HYPOTHETICAL, epsilon=ALPHA)

    # -- composition --------------------------------------------------------
    @staticmethod
    def compose_path(relators: Sequence[Relator]) -> Optional[Relator]:
        """Fold-compose R₁∘R₂∘…∘R_K (MATH_SPEC §1.2); None if any hop fails."""
        if not relators:
            return None
        acc = relators[0]
        for nxt in relators[1:]:
            acc = acc.compose(nxt)
            if acc is None:
                return None
        return acc

    # -- §6 uncertainty -----------------------------------------------------
    def chain_uncertainty(self, path: Sequence[Relator]) -> float:
        """σ_conclusion lower bound = (min_σ)^K · exp(−λ·Σ ΔS_i) (MATH_SPEC §6)."""
        if not path:
            return 0.0
        k = len(path)
        min_sigma = min(r.sigma for r in path)
        sum_surprise = sum(r.epsilon for r in path)
        return (min_sigma**k) * math.exp(-self.lam * sum_surprise)

    # -- the sparse inference (MATH_SPEC §6 / the TORIS spec §3.7) --------------
    def infer_along(
        self,
        field: RelationalField,
        concept_sequence: Sequence[ConceptState],
        tau_default: RelationType = RelationType.CAUSAL,
    ) -> ChainResult:
        """Walk an expected K-hop route, bridging gaps with hypotheticals.

        For each consecutive pair (cᵢ, cᵢ₊₁): use the strongest existing relator
        on that directed edge as a seed, or instantiate a hypothetical connector
        if none exists. The chain is ``broken`` only if an endpoint concept is
        missing entirely. Returns a ChainResult with both the realized strength
        (Π σ_i) and the §6 lower bound.
        """
        path: List[Relator] = []
        n_seed = 0
        n_hypothetical = 0
        broken = False

        for src, tgt in zip(concept_sequence, concept_sequence[1:]):
            if src is None or tgt is None:
                broken = True
                break
            candidates = field.relators_between(src, tgt)
            if candidates:
                seed = max(candidates, key=lambda r: r.sigma)
                path.append(seed)
                n_seed += 1
            else:
                hyp = self.instantiate_hypothetical(src, tgt, tau_default)
                field.add_relator(hyp)
                path.append(hyp)
                n_hypothetical += 1

        if broken or not path:
            return ChainResult(
                path=path,
                hops=len(path),
                n_seed=n_seed,
                n_hypothetical=n_hypothetical,
                min_sigma=0.0,
                sum_surprise=sum(r.epsilon for r in path),
                sigma_chain=0.0,  # broken chain → collapses to zero
                sigma_bound=0.0,
                broken=True,
                composite=None,
            )

        sigma_chain = 1.0
        for r in path:
            sigma_chain *= r.sigma
        return ChainResult(
            path=path,
            hops=len(path),
            n_seed=n_seed,
            n_hypothetical=n_hypothetical,
            min_sigma=min(r.sigma for r in path),
            sum_surprise=sum(r.epsilon for r in path),
            sigma_chain=sigma_chain,
            sigma_bound=self.chain_uncertainty(path),
            broken=False,
            composite=self.compose_path(path),
        )
