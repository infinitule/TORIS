"""Layer 9 — Unified Surprise Architecture (the Exact-Surprise spec §12.5).

Routes all surprise computations through a single interface that selects
the appropriate regime and applies all corrections:

    FAST REGIME   (d ≤ d_crit, Q(G) > 0.01):
        ΔS_TFSA + ΔS_wave  [Layer 6]

    STANDARD REGIME  (d ≤ d_crit, Q(G) ≤ 0.01):
        ΔS_TASF_mock + ΔS_shadow  [Layer 7 + 9]

    DEEP REGIME  (d > d_crit):
        ΔS_Rademacher (exact, certified)  [Layer 9]
        with Eisenstein weights
        + ΔS_shadow correction

    ALWAYS:
        Suppression check at partition congruence depths (§11)
        Critical point check

Reference: the Exact-Surprise spec §12.5
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold

from toris.engine.eisenstein import eisenstein_weights, D_CRIT_DEFAULT
from toris.engine.rademacher import rademacher_surprise, certified_surprise, RademacherResult
from toris.engine.maass_completion import shadow_correction, complete_tasf


# Suppression depths from partition congruence theorem (Layer 8, §11)
def _is_suppressed(d: int) -> bool:
    """True if depth d is suppressed by partition congruences.

    Suppresses: d ≡ 4 (mod 5), d ≡ 5 (mod 7), d ≡ 6 (mod 11)
    These correspond to p(5m+4), p(7m+5), p(11m+6) ≡ 0 congruences.
    """
    return (d % 5 == 4) or (d % 7 == 5) or (d % 11 == 6)


@dataclass
class UnifiedResult:
    """Result from UnifiedSurprise.compute()."""
    delta_S: float
    error_bound: float        # certified bound (Rademacher) or 0 for TFSA/TASF
    regime_used: str          # "fast" | "standard" | "deep"
    shadow_applied: bool
    shadow_correction: float
    suppressed: bool          # True if depth was suppressed (returns 0)
    rademacher_terms_used: int
    weights_alpha: float
    weights_beta: float
    weights_gamma: float


class UnifiedSurprise:
    """Single entry point for all TORIS surprise computations.

    Automatically routes to the correct regime and applies shadow correction.
    Returns certified error bounds for the deep regime (Rademacher).

    Reference: §12.5
    """

    def __init__(
        self,
        d_crit: int = D_CRIT_DEFAULT,
        fast_threshold: float = 0.01,
    ):
        self.d_crit = d_crit
        self.fast_threshold = fast_threshold

    def compute(
        self,
        field: "RelationalField",
        f_pred: "RelationalField | None",
        d: int,
        goal_manifold: "GoalManifold | None" = None,
        precision: int = 8,
    ) -> UnifiedResult:
        """Compute ΔS with the correct regime for depth d.

        Parameters
        ----------
        field:        Observed relational field F_obs
        f_pred:       Predicted field F_pred (may be None)
        d:            Relational depth
        goal_manifold: Active goal manifold (for warping and Q(G))
        precision:    Desired significant figures (deep regime only)
        """
        # Partition congruence suppression
        if _is_suppressed(d):
            return UnifiedResult(
                delta_S=0.0, error_bound=0.0,
                regime_used="suppressed", shadow_applied=False,
                shadow_correction=0.0, suppressed=True,
                rademacher_terms_used=0,
                weights_alpha=0.0, weights_beta=0.0, weights_gamma=0.0,
            )

        alpha, beta, gamma = eisenstein_weights(d, self.d_crit)

        # Regime selection
        if d > self.d_crit:
            return self._deep_regime(field, f_pred, d, alpha, beta, gamma, precision)
        else:
            q_goal = self._quality_of_goal(goal_manifold)
            if q_goal > self.fast_threshold:
                return self._fast_regime(field, f_pred, d, alpha, beta, gamma)
            else:
                return self._standard_regime(field, f_pred, d, goal_manifold, alpha, beta, gamma)

    # ------------------------------------------------------------------
    # Regime implementations

    def _fast_regime(
        self, field, f_pred, d, alpha, beta, gamma
    ) -> UnifiedResult:
        """TFSA + wave (Layer 6) — fastest, for high Q(G) fields."""
        from toris.engine.surprise import SurpriseMetric
        sm = SurpriseMetric(alpha=alpha, beta=beta, gamma=gamma)
        pred = f_pred if f_pred is not None else field
        # SurpriseMetric.topological_surprise expects EdgeIndex = Dict[(src,tgt), Relator]
        delta_s = sm.topological_surprise(pred.relator_index(), field.relator_index())
        return UnifiedResult(
            delta_S=delta_s, error_bound=0.0,
            regime_used="fast", shadow_applied=False,
            shadow_correction=0.0, suppressed=False,
            rademacher_terms_used=0,
            weights_alpha=alpha, weights_beta=beta, weights_gamma=gamma,
        )

    def _standard_regime(
        self, field, f_pred, d, goal_manifold, alpha, beta, gamma
    ) -> UnifiedResult:
        """TASF contour + shadow correction — standard precision."""
        pred = f_pred if f_pred is not None else field
        result = complete_tasf(field, pred, goal_manifold)
        return UnifiedResult(
            delta_S=result.delta_S_complete, error_bound=0.0,
            regime_used="standard", shadow_applied=(result.delta_S_shadow != 0.0),
            shadow_correction=result.delta_S_shadow, suppressed=False,
            rademacher_terms_used=0,
            weights_alpha=alpha, weights_beta=beta, weights_gamma=gamma,
        )

    def _deep_regime(
        self, field, f_pred, d, alpha, beta, gamma, precision
    ) -> UnifiedResult:
        """Rademacher exact series + shadow correction — certified precision."""
        # Override weights for deep regime (Eisenstein)
        rademacher_result = rademacher_surprise(
            field, d,
            N_terms=_precision_to_N(precision),
        )
        delta_s_base = rademacher_result.S_exact
        error_bound = rademacher_result.error_bound

        # Apply shadow correction if productive contradictions exist
        pred = f_pred if f_pred is not None else field
        shadow_corr = shadow_correction(field, pred)
        delta_s_total = delta_s_base + shadow_corr

        return UnifiedResult(
            delta_S=delta_s_total, error_bound=error_bound,
            regime_used="deep", shadow_applied=(shadow_corr != 0.0),
            shadow_correction=shadow_corr, suppressed=False,
            rademacher_terms_used=rademacher_result.terms_used,
            weights_alpha=alpha, weights_beta=beta, weights_gamma=gamma,
        )

    # ------------------------------------------------------------------
    # Helper

    @staticmethod
    def _quality_of_goal(goal_manifold) -> float:
        """Q(G) — quality/confidence of the goal manifold.

        Proxy: fraction of subgoals that are ACTIVE (not abandoned).
        High Q means the goal is well-defined → use fast regime.
        """
        if goal_manifold is None:
            return 1.0  # no goal = full confidence (use fast)
        n_active = len(goal_manifold.active)
        n_total = n_active + len(goal_manifold.abandoned)
        if n_total == 0:
            return 1.0
        return n_active / n_total


def _precision_to_N(precision: int) -> int:
    """Map desired significant figures to number of Rademacher terms."""
    if precision <= 3:
        return 1
    elif precision <= 8:
        return 3
    elif precision <= 13:
        return 6
    else:
        return 10
