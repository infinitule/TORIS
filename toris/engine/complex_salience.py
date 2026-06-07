"""Layer 7 — Complex salience field and surprise density F(κ).

Extends real κ ∈ [0,1] to the complex disk |κ| ≤ κ_max.
Mirrors the tau-physics correlator Π(s) in complex momentum-squared space.
"""

from __future__ import annotations

import math
import cmath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold

from toris.primitives.relation_types import d_type
from toris.goal.warp import relevance as _relevance


def encode_complex(kappa_real: float, phase: float = 0.0) -> complex:
    """Map real κ ∈ [0,1] to complex κ on the unit disk.

    Default phase=0 gives the real axis; non-zero phase rotates off-axis.
    """
    kappa_real = max(0.0, min(1.0, kappa_real))
    return kappa_real * cmath.exp(1j * phase)


class ComplexSalienceField:
    """Evaluates the surprise density F(κ) at complex salience κ.

    F(κ) = F_directed(κ) + W_goal(κ, G) · F_undirected(κ)

    Analytic in the open disk |κ| < κ_max; discontinuities on the real
    interval [0, κ_max] correspond to productive contradictions (poles).
    """

    def __init__(self, kappa_max: float = 1.0):
        self.kappa_max = kappa_max

    # ------------------------------------------------------------------
    # Component densities

    def F_directed(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        kappa_complex: complex,
    ) -> complex:
        """Directed surprise density from typed relator mismatches.

        F^(dir)(κ) = Σ_R σ(R) · [1 - match(τ_pred, τ_obs)] · κ(R)
                     evaluated at complex κ via analytic extension.
        """
        total: complex = 0.0 + 0j

        pred_index = f_pred.relator_index()   # (src,tgt) → strongest Relator
        obs_index  = field.relator_index()

        # Threshold above which a type mismatch is treated as a genuine
        # contradiction — adds a pole term 1/(κ − κ_C) to F(κ).
        # d_type returns 0.7 for CONTRA pairs and 1.0 for direct CONTRADICTS.
        POLE_THRESHOLD = 0.65

        # Matched edges: smooth type-mismatch contribution + pole terms
        for edge, r_obs in obs_index.items():
            r_pred = pred_index.get(edge)
            if r_pred is not None:
                type_mismatch = d_type(r_pred.tau, r_obs.tau)   # ∈ [0,1]
                # Smooth part: analytic extension replacing real κ with κ_complex
                kappa_factor = r_obs.kappa * kappa_complex / self.kappa_max
                total += r_obs.sigma * type_mismatch * kappa_factor

                # Pole term for genuine contradictions (PRODUCTIVE contradictions
                # = poles in F(κ), §10.2.4).
                # Adds: Res / (κ_complex − κ_C) where Res = σ_obs·σ_pred·d_type.
                if type_mismatch >= POLE_THRESHOLD:
                    kappa_pole = r_obs.kappa  # real-axis pole location
                    res = r_obs.sigma * r_pred.sigma * type_mismatch
                    denom = kappa_complex - kappa_pole
                    if abs(denom) > 1e-20:
                        total += res / denom

        return total

    def F_undirected(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        kappa_complex: complex,
    ) -> complex:
        """Structural gap surprise density.

        F^(und)(κ) = |E_obs \\ E_pred| · κ_avg at complex κ.
        """
        pred_edges = f_pred.edge_set()
        obs_edges  = field.edge_set()

        new_edges = obs_edges - pred_edges
        n_new = len(new_edges)
        if n_new == 0:
            return 0.0 + 0j

        kappa_avg = 0.0
        all_relators = list(field.relators())
        if all_relators:
            kappa_avg = sum(r.kappa for r in all_relators) / len(all_relators)

        kappa_factor = kappa_avg * kappa_complex / self.kappa_max
        return complex(n_new) * kappa_factor

    def W_goal(
        self,
        kappa_complex: complex,
        goal_manifold: GoalManifold | None,
    ) -> complex:
        """Goal manifold warp factor at complex κ.

        W_goal(κ, G) = Π_{g∈G_active} [1 + priority(g) · |κ|/κ_max]

        Analogous to SEW = 1.0194 electroweak correction in tau physics.
        """
        if goal_manifold is None or not goal_manifold.active:
            return 1.0 + 0j

        scale = abs(kappa_complex) / self.kappa_max
        result: complex = 1.0 + 0j
        for subgoal in goal_manifold.active:
            result *= 1.0 + subgoal.priority * scale
        return result

    def surprise_density_F(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        kappa_complex: complex,
        goal_manifold: GoalManifold | None = None,
    ) -> complex:
        """Full surprise density F(κ) at complex salience κ.

        F(κ) = F_dir(κ) + W_goal(κ,G) · F_und(κ)
        """
        f_dir = self.F_directed(field, f_pred, kappa_complex)
        f_und = self.F_undirected(field, f_pred, kappa_complex)
        w     = self.W_goal(kappa_complex, goal_manifold)
        return f_dir + w * f_und
