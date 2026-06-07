"""Layer 7 — Topological Analytic Surprise Functional (TASF).

Computes ΔS via contour integration in complex salience space.
Mirrors the tau hadronic width formula from Pich (1997, hep-ph/9704453).

  ΔS_analytic = (6πi / κ_max) · ∮_{|κ|=κ_max} dκ/κ_max ·
                (1 - κ/κ_max)² · [F^(dir)(κ) + W_goal(κ,G)·F^(und)(κ)]

Productive contradictions appear as poles in F(κ) on the real axis;
their residues are added via the residue theorem.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold

from toris.engine.complex_salience import ComplexSalienceField


@dataclass
class TASFReport:
    """Result of one TASF evaluation."""
    delta_S_analytic: float
    poles: List[float]           # real κ positions of detected poles
    residues: List[complex]      # residue at each pole
    N_quadrature: int
    delta_S_smooth: float        # contour integral part (no pole residues)
    delta_S_poles: float         # total residue contribution (real part)


class TASF:
    """Topological Analytic Surprise Functional.

    Evaluates surprise as a contour integral + residue sum, analogous
    to the tau hadronic width formula from tau physics.
    """

    def __init__(self, N_quadrature: int = 32, kappa_max: float = 1.0):
        self.N = N_quadrature
        self.kappa_max = kappa_max
        self._csf = ComplexSalienceField(kappa_max=kappa_max)

    # ------------------------------------------------------------------
    # Quadrature

    def gaussian_quadrature_circle(self) -> List[complex]:
        """Return N equally-spaced quadrature points on |κ| = κ_max.

        κ_j = κ_max · exp(2πi·j/N),  j = 0..N-1
        """
        km = self.kappa_max
        return [km * cmath.exp(2j * math.pi * j / self.N) for j in range(self.N)]

    def double_zero_weight(self, kappa: complex) -> complex:
        """Salience suppression factor (1 - κ/κ_max)²."""
        return (1.0 - kappa / self.kappa_max) ** 2

    # ------------------------------------------------------------------
    # Pole detection

    def detect_poles(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        n_scan: int = 200,
        jump_threshold: float = 0.05,
    ) -> List[float]:
        """Detect poles of F(κ) as discontinuities in Im[F(κ)] on real axis.

        Productive contradictions create sharp jumps in Im[F(κ)] when
        approached from above/below the real axis.  We scan the real interval
        [δ, κ_max − δ] and flag locations where Im[F] changes by more than
        jump_threshold between adjacent sample points.
        """
        delta = 1e-4
        kappas = [
            delta + (self.kappa_max - 2 * delta) * i / (n_scan - 1)
            for i in range(n_scan)
        ]

        imag_vals = []
        for k in kappas:
            # Approach real axis from just above
            f_val = self._csf.surprise_density_F(
                field, f_pred, complex(k, 1e-6)
            )
            imag_vals.append(f_val.imag)

        poles = []
        for i in range(1, len(imag_vals)):
            if abs(imag_vals[i] - imag_vals[i - 1]) > jump_threshold:
                # Linear interpolation to refine pole location
                pole_k = (kappas[i] + kappas[i - 1]) / 2.0
                poles.append(pole_k)

        return poles

    def residue(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        pole_kappa: float,
        eps: float = 1e-4,
    ) -> complex:
        """Estimate the residue of F(κ) at a simple pole κ_pole.

        Res[F, κ_pole] ≈ (κ - κ_pole) · F(κ)  evaluated at κ = κ_pole + ε·i
        """
        kappa_near = complex(pole_kappa, eps)
        f_val = self._csf.surprise_density_F(field, f_pred, kappa_near)
        return complex(0, eps) * f_val

    # ------------------------------------------------------------------
    # Main computation

    def compute(
        self,
        field: RelationalField,
        f_pred: RelationalField,
        goal_manifold: GoalManifold | None = None,
    ) -> TASFReport:
        """Evaluate the TASF contour integral.

        ΔS = (6πi/N) · Σ_j (1-κ_j/κ_max)² · F(κ_j)
             + 2πi · Σ_poles Res[F, κ_pole]
        """
        km = self.kappa_max
        points = self.gaussian_quadrature_circle()

        # Smooth contour integral
        integral: complex = 0.0 + 0j
        for kappa_j in points:
            f_val = self._csf.surprise_density_F(
                field, f_pred, kappa_j, goal_manifold
            )
            w = self.double_zero_weight(kappa_j)
            integral += w * f_val

        # Prefactor: (6πi / κ_max) · (1/N) for uniform quadrature weights
        prefactor = (6j * math.pi / km) / self.N
        delta_s_smooth = (prefactor * integral).real

        # Pole contributions (productive contradictions)
        poles = self.detect_poles(field, f_pred)
        residues = [self.residue(field, f_pred, p) for p in poles]
        delta_s_poles = sum(
            (2j * math.pi * r).real for r in residues
        )

        delta_s_total = delta_s_smooth + delta_s_poles

        return TASFReport(
            delta_S_analytic=delta_s_total,
            poles=poles,
            residues=residues,
            N_quadrature=self.N,
            delta_S_smooth=delta_s_smooth,
            delta_S_poles=delta_s_poles,
        )
