"""Layer 7 — TORIS Michel Parameters.

Typed relation deviation detectors analogous to the Michel parameters
(ρ, η, ξ, δ) in tau decay.  Standard TORIS values:

  ρ_T = 3/4   (structural confirmation)
  η_T = 0     (no systematic type confusion)
  ξ_T = 1     (goal-aligned surprise dominates)
  δ_T ≈ 0.01  (non-perturbative effects small)

Deviations trigger a Michel Alert → switch to full contour integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold

from toris.primitives.relation_types import d_type
from toris.engine.relational_ope import RelationalOPE

# Michel Alert bounds (§10.4.3)
_BOUNDS = {
    "rho_T": (3 / 4 - 0.05, 3 / 4 + 0.05),
    "eta_T": (-0.05, 0.05),
    "xi_T": (0.9, float("inf")),
    "delta_T": (0.0, 0.05),
}


@dataclass
class MichelParameters:
    """TORIS Michel Parameters (ρ_T, η_T, ξ_T, δ_T)."""
    rho_T: float
    eta_T: float
    xi_T: float
    delta_T: float

    def in_bounds(self) -> bool:
        """True iff all parameters are within standard bounds."""
        lo, hi = _BOUNDS["rho_T"]
        if not (lo <= self.rho_T <= hi):
            return False
        lo, hi = _BOUNDS["eta_T"]
        if not (lo <= self.eta_T <= hi):
            return False
        lo, hi = _BOUNDS["xi_T"]
        if not (self.xi_T >= lo):
            return False
        lo, hi = _BOUNDS["delta_T"]
        if not (lo <= self.delta_T <= hi):
            return False
        return True


def standard_values() -> MichelParameters:
    """Standard TORIS Michel parameter values for a well-calibrated field."""
    return MichelParameters(rho_T=3 / 4, eta_T=0.0, xi_T=1.0, delta_T=0.01)


def michel_alert(params: MichelParameters) -> bool:
    """True if any parameter is outside standard bounds (§10.4.3).

    A Michel Alert means the inference chain is operating in a regime where
    the standard perturbative surprise calculation is unreliable.
    """
    return not params.in_bounds()


def compute(
    field: RelationalField,
    f_pred: RelationalField,
    goal: GoalManifold | None = None,
) -> MichelParameters:
    """Compute TORIS Michel parameters for (field, f_pred) pair.

    Parameters
    ----------
    field:  observed relational field
    f_pred: predicted relational field
    goal:   active goal manifold (used for ξ_T directionality)
    """
    pred_edges = f_pred.edge_set()
    obs_edges  = field.edge_set()
    matched    = pred_edges & obs_edges

    n_pred   = max(len(pred_edges), 1)
    n_match  = max(len(matched),    1)

    # ρ_T — structural confirmation ratio
    rho_T = (len(matched) / n_pred) * (3 / 4)

    # η_T — average type mismatch on matched edges
    pred_index = f_pred.relator_index()
    obs_index  = field.relator_index()
    type_mismatches = []
    for edge in matched:
        r_p = pred_index.get(edge)
        r_o = obs_index.get(edge)
        if r_p is not None and r_o is not None:
            type_mismatches.append(d_type(r_p.tau, r_o.tau))
    eta_T = (0.5 * sum(type_mismatches) / len(type_mismatches)
             if type_mismatches else 0.0)

    # ξ_T — directional asymmetry
    # "forward" = relators with ε > θ that are goal-relevant
    # "backward" = relators with ε > θ that are not goal-relevant
    from toris.constants import THETA_EPSILON
    goal_concepts: set[str] = set()
    if goal is not None:
        goal_concepts = goal.primary.concepts | {
            c for sg in goal.active for c in sg.concepts
        }

    forward_eps = 0.0
    backward_eps = 0.0
    for r in field.relators():
        if r.epsilon > THETA_EPSILON:
            if r.src_id in goal_concepts or r.tgt_id in goal_concepts:
                forward_eps += r.epsilon
            else:
                backward_eps += r.epsilon

    total_dir = forward_eps + backward_eps
    if total_dir > 0:
        xi_T = (forward_eps - backward_eps) / total_dir
    else:
        xi_T = 1.0   # no propagating relators → standard value

    # δ_T — deep structure ratio: C_4·S_4 / (C_0·S_0)
    ope = RelationalOPE()
    expansion = ope.expand(field, max_depth=4)
    s0 = expansion.condensates.get(0, 0.0)
    s4 = expansion.condensates.get(4, 0.0)
    c0 = expansion.coefficients.get(0, 1.0)
    c4 = expansion.coefficients.get(4, 0.01)
    denom = c0 * s0
    delta_T = (c4 * s4 / denom) if denom > 1e-10 else 0.0

    return MichelParameters(rho_T=rho_T, eta_T=eta_T, xi_T=xi_T, delta_T=delta_T)
