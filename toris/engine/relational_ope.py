"""Layer 7 — Relational Operator Product Expansion (OPE).

Multi-scale surprise decomposition by relational depth, analogous to the
OPE expansion of Π(s) in powers of 1/s in tau physics:

  F(κ) = Σ_{d=0,2,4,...} C_d(κ) · <S_d> / κ^(d/2)

  d=0: local per-relator type mismatch       (perturbative)
  d=2: one-hop neighborhood average          (quark-mass analog)
  d=4: loop surprise from wave dynamics      (non-perturbative)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

from toris.primitives.relation_types import d_type
from toris.engine.wave import CyclicWaveEngine


@dataclass
class OPEExpansion:
    """Result of Relational OPE expansion."""
    condensates: Dict[int, float]    # S_d: condensate at each depth d
    coefficients: Dict[int, float]   # C_d: Wilson coefficient at each depth
    depths: List[int]                # [0, 2, 4]

    def total(self, kappa_ref: float = 0.5) -> float:
        """Sum OPE: Σ_d C_d · S_d / κ^(d/2)."""
        total = 0.0
        for d in self.depths:
            denom = kappa_ref ** (d / 2.0) if d > 0 else 1.0
            total += self.coefficients[d] * self.condensates[d] / denom
        return total


class RelationalOPE:
    """Computes the Relational OPE for the surprise density.

    Organises surprise contributions by relational depth, providing a
    perturbative expansion analogous to the QCD OPE.
    """

    # Wilson coefficients from tau physics analog (§10.3)
    C_COEFFICIENTS = {0: 1.0, 2: 0.1, 4: 0.01}

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self._wave_engine = CyclicWaveEngine()

    # ------------------------------------------------------------------
    # Condensate extractors

    def depth_0_surprise(self, field: RelationalField) -> float:
        """d=0: local per-relator type mismatch (perturbative, O(1))."""
        relators = list(field.relators())
        if not relators:
            return 0.0
        return sum(r.epsilon for r in relators) / len(relators)

    def depth_2_surprise(self, field: RelationalField) -> float:
        """d=2: one-hop neighborhood average (quark-mass analog).

        For each relator R, average surprise in its 1-hop neighborhood
        (relators sharing a concept endpoint with R).
        """
        relators = list(field.relators())
        if not relators:
            return 0.0

        total = 0.0
        for r in relators:
            nbr_src = field.get_neighborhood(r.src, depth=1)
            nbr_tgt = field.get_neighborhood(r.tgt, depth=1)
            # get_neighborhood returns a RelationalField; extract concept ids
            nbr_ids = {c.id for c in nbr_src.concepts()} | {c.id for c in nbr_tgt.concepts()}
            nbr_relators = [
                nbr_r for nbr_r in field.relators()
                if nbr_r.rid != r.rid and (
                    nbr_r.src_id in nbr_ids or nbr_r.tgt_id in nbr_ids
                )
            ]
            if nbr_relators:
                total += sum(nr.epsilon for nr in nbr_relators) / len(nbr_relators)
            else:
                total += r.epsilon

        return total / len(relators)

    def depth_4_surprise(self, field: RelationalField) -> float:
        """d=4: loop surprise (non-perturbative, from wave dynamics §9.2).

        Uses the cyclic wave engine to identify unstable loops and measure
        the mean final surprise amplitude as the d=4 condensate.
        """
        unstable = self._wave_engine.scan_field(field)
        if not unstable:
            return 0.0
        all_eps = [
            eps for cycle_result in unstable
            for eps in cycle_result.final_epsilons
        ]
        return sum(all_eps) / len(all_eps) if all_eps else 0.0

    # ------------------------------------------------------------------
    # Spectral moments

    def spectral_moments(
        self,
        field: RelationalField,
        k_values: List[int],
        l_values: List[int],
        kappa_0: float = 1.0,
        n_points: int = 50,
    ) -> Dict[Tuple[int, int], float]:
        """Compute spectral moments M^kl(κ_0).

        M^kl(κ_0) = ∫_0^{κ_0} dκ · (1-κ/κ_0)^k · (κ/κ_max)^l · dΔS/dκ

        Approximated by finite differences over n_points samples.
        """
        kappa_max = 1.0
        kappas = [kappa_0 * i / n_points for i in range(1, n_points + 1)]
        dk = kappa_0 / n_points

        # Approximate dΔS/dκ using relator density weighted by κ
        # ΔS(κ) ≈ Σ_R ε(R) · [κ(R) ≤ κ]
        relators = list(field.relators())

        moments: Dict[Tuple[int, int], float] = {}
        for k in k_values:
            for l in l_values:
                total = 0.0
                for i, kappa in enumerate(kappas):
                    # dΔS/dκ at this point: relators with kappa(R) ≈ this κ
                    eps_at_kappa = sum(
                        r.epsilon
                        for r in relators
                        if abs(r.kappa - kappa) < dk
                    )
                    w_k = (1.0 - kappa / kappa_0) ** k
                    w_l = (kappa / kappa_max) ** l
                    total += w_k * w_l * eps_at_kappa * dk
                moments[(k, l)] = total

        return moments

    # ------------------------------------------------------------------
    # Main expand

    def expand(
        self,
        field: RelationalField,
        max_depth: int | None = None,
    ) -> OPEExpansion:
        """Compute the OPE expansion up to max_depth."""
        max_depth = max_depth if max_depth is not None else self.max_depth
        depths = [d for d in [0, 2, 4] if d <= max_depth]

        condensates: Dict[int, float] = {}
        if 0 in depths:
            condensates[0] = self.depth_0_surprise(field)
        if 2 in depths:
            condensates[2] = self.depth_2_surprise(field)
        if 4 in depths:
            condensates[4] = self.depth_4_surprise(field)

        coefficients = {d: self.C_COEFFICIENTS.get(d, 0.0) for d in depths}

        return OPEExpansion(
            condensates=condensates,
            coefficients=coefficients,
            depths=depths,
        )
