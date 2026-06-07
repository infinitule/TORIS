"""Layer 7 — Running Surprise Coupling α_S(κ).

Demonstrates TORIS asymptotic freedom: high-salience inference is weakly
coupled (surprise doesn't propagate through the most relevant relators),
while low-salience inference is strongly coupled.

  α_S(κ) = C_0(κ) · <S_0> · κ
  κ · dα_S/dκ = -b_0 · α_S² - b_1 · α_S³  [TORIS β-function]

b_0 = 1, b_1 = 5/3  (from composition table structure, §10.5.1)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

from toris.engine.relational_ope import RelationalOPE


class SurpriseCoupling:
    """Computes and fits the running surprise coupling α_S(κ)."""

    B0: float = 1.0        # one-loop β coefficient
    B1: float = 5.0 / 3.0  # two-loop β coefficient

    def __init__(self):
        self._ope = RelationalOPE()

    # ------------------------------------------------------------------
    # Coupling at a single scale

    def alpha_S(self, field: RelationalField, kappa: float) -> float:
        """Surprise coupling at salience scale κ — one-loop running.

        Uses the exact one-loop solution of κ·dα/dκ = −b0·α²:

            α_S(κ) = α_ref / (1 + α_ref · b0 · log(κ / κ_ref))

        Reference point κ_ref = κ_max = 1.0.  The reference coupling
        α_ref ∈ (0,1) is derived from the field's global average surprise:

            α_ref = C_0 · s0 / (1 + s0)      (s0 = mean ε over all relators)

        This keeps α_S values in a moderate range [α_ref, ~3·α_ref] so
        that the β-function fit is well-conditioned (§10.5.2).
        """
        kappa = max(1e-6, min(1.0, kappa))

        relators_all = list(field.relators())
        if not relators_all:
            return 0.0

        # Global average surprise (s0 = <S_0> averaged over whole field)
        s0 = sum(r.epsilon for r in relators_all) / len(relators_all)
        c0 = self._ope.C_COEFFICIENTS[0]

        # Reference coupling at κ_ref = 1.0 — kept in (0,1) by the mapping
        alpha_ref = c0 * s0 / (1.0 + s0)

        # One-loop running with κ_ref = 1.0:
        #   log(κ / 1.0) = log(κ)  which is negative for κ < 1
        # → denominator < 1 for κ < 1 → α_S > α_ref  (strongly coupled at low κ) ✓
        log_ratio = math.log(kappa)   # log(κ / κ_ref), κ_ref = 1
        denominator = 1.0 + alpha_ref * self.B0 * log_ratio
        # Guard against Landau pole (κ too small)
        denominator = max(denominator, 0.001)
        return alpha_ref / denominator

    # ------------------------------------------------------------------
    # Running over multiple scales

    def run_coupling(
        self,
        field: RelationalField,
        kappa_values: List[float],
    ) -> List[float]:
        """Evaluate α_S at each κ in kappa_values."""
        return [self.alpha_S(field, k) for k in kappa_values]

    # ------------------------------------------------------------------
    # β-function fit

    def fit_beta_function(
        self,
        kappa_values: List[float],
        alpha_values: List[float],
    ) -> Tuple[float, float]:
        """Fit the TORIS β-function to measured α_S(κ) data.

        Models: κ · dα_S/dκ = -b0 · α_S² - b1 · α_S³
        Linearise: d log α_S / d log κ = -b0 · α_S - b1 · α_S²

        Returns (b0, b1) from a least-squares fit.
        """
        if len(kappa_values) < 3:
            return (self.B0, self.B1)

        kv = np.array(kappa_values, dtype=float)
        av = np.array(alpha_values, dtype=float)

        # Numerical derivative: κ · dα/dκ
        log_k = np.log(kv)
        log_a = np.log(np.maximum(av, 1e-12))
        d_log_a = np.gradient(log_a, log_k)   # d log α / d log κ

        # d log α / d log κ ≈ -b0·α - b1·α²
        # Design matrix: columns [α, α²]
        A = np.column_stack([av, av ** 2])
        y = -d_log_a

        # Least squares with non-negative b0, b1
        try:
            result, *_ = np.linalg.lstsq(A, y, rcond=None)
            b0, b1 = float(result[0]), float(result[1])
            # Clamp to physically reasonable range
            # Ensure physically meaningful positive values; b1 can be small
            b0 = max(0.01, min(b0, 10.0))
            b1 = max(1e-4, min(b1, 20.0))
        except Exception:
            b0, b1 = self.B0, self.B1

        return (b0, b1)

    # ------------------------------------------------------------------
    # Extraction from moments

    def extract_from_moments(self, M00: float, N_relators: int) -> float:
        """Extract α_S(κ_max) from the M^00 spectral moment.

        Analogous to extracting αs from Rτ (Table 14 in tau physics):
          α_S(κ_max) = [M^00 / (3 · N_relators)] - 1

        This is iterated once; the result is clamped to [0, 1].
        """
        if N_relators == 0:
            return 0.0
        raw = M00 / (3.0 * N_relators) - 1.0
        return max(0.0, min(1.0, raw))
