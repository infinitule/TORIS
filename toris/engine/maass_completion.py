"""Layer 9 — Harmonic Maass Shadow Completion (the Exact-Surprise spec §12.3).

Computes the non-holomorphic shadow correction ΔS_shadow for fields with
productive contradictions.

Without this correction, TASF underestimates surprise in contradiction-rich fields.
The shadow F^-(κ, κ̄) is the Eichler integral of the shadow cusp form g_C(z).

    ΔS = ΔS_mock + ΔS_shadow
       = ΔS_TASF + Σ_C ∮ F^-_C(κ,κ̄) · (1−κ/κ_max)² · dκ̄

For a field with one productive contradiction of strength σ=0.7:
    |ΔS_shadow| ≈ 2π · 0.49 ≈ 3.08

Reference: the Exact-Surprise spec §12.3
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

from toris.engine.rademacher import TAU_INDEX
from toris.reasoning.contradiction import ContradictionLog, ContradictionEntry, ResolutionStatus

try:
    from scipy.integrate import quad
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


@dataclass
class ShadowContradiction:
    """A productive contradiction with its shadow parameters."""
    relator_a_sigma: float
    relator_b_sigma: float
    kappa_C: float          # real-axis pole location (average kappa of the pair)
    tau_diff: int           # |τ_index(R_a) − τ_index(R_b)|
    residue_strength: float # |Res[F^+, κ_C]| = σ_a · σ_b · type_distance


@dataclass
class CompleteResult:
    """Result of the complete TASF with Maass shadow correction."""
    delta_S_mock: float
    delta_S_shadow: float
    delta_S_complete: float
    shadow_fraction: float   # |ΔS_shadow| / |ΔS_complete|


def shadow_cusp_form(contradiction: ShadowContradiction, z: complex) -> complex:
    """g_C(z) — shadow cusp form of contradiction C.

        g_C(z) = σ(R_a) · σ(R_b) · exp(2πi · τ_diff(C) · z)

    Reference: §12.3.3
    """
    return (
        contradiction.relator_a_sigma
        * contradiction.relator_b_sigma
        * cmath.exp(2j * math.pi * contradiction.tau_diff * z)
    )


def eichler_integral(
    contradiction: ShadowContradiction,
    kappa: complex,
    kappa_max: float = 1.0,
) -> complex:
    """E_C(κ, κ̄) — Eichler integral of the shadow cusp form.

        E_C(κ, κ̄) = ∫_{−κ̄}^{κ_max} g_C(z) · (z + κ)^(−2) dz

    Evaluated numerically via scipy.integrate.quad with complex integrand.
    Reference: §12.3.3
    """
    kappa_bar = kappa.conjugate()
    lower = -kappa_bar.real  # real part of −κ̄
    upper = kappa_max

    if not _HAS_SCIPY or upper <= lower:
        # Fallback: trapezoidal approximation
        N = 100
        ts = [lower + (upper - lower) * i / N for i in range(N + 1)]
        result: complex = 0.0 + 0j
        for i in range(N):
            t = (ts[i] + ts[i + 1]) / 2.0
            z = complex(t, 0.0)
            denom = (z + kappa) ** 2
            if abs(denom) > 1e-20:
                result += shadow_cusp_form(contradiction, z) / denom
        return result * (upper - lower) / N

    # scipy path: split into real and imaginary parts
    def integrand_real(t: float) -> float:
        z = complex(t, 0.0)
        denom = (z + kappa) ** 2
        if abs(denom) < 1e-20:
            return 0.0
        val = shadow_cusp_form(contradiction, z) / denom
        return val.real

    def integrand_imag(t: float) -> float:
        z = complex(t, 0.0)
        denom = (z + kappa) ** 2
        if abs(denom) < 1e-20:
            return 0.0
        val = shadow_cusp_form(contradiction, z) / denom
        return val.imag

    try:
        re_val, _ = quad(integrand_real, lower, upper, limit=100)
        im_val, _ = quad(integrand_imag, lower, upper, limit=100)
        return complex(re_val, im_val)
    except Exception:
        return 0.0 + 0j


def shadow_density(
    contradiction: ShadowContradiction,
    kappa: complex,
    kappa_max: float = 1.0,
) -> complex:
    """F^-_C(κ, κ̄) — shadow surprise density of contradiction C.

        F^-_C(κ, κ̄) = Res[F^+, κ_C] · E_C(κ, κ̄)

    Reference: §12.3.3
    """
    res = contradiction.residue_strength
    E = eichler_integral(contradiction, kappa, kappa_max)
    return res * E


def shadow_correction(
    field: "RelationalField",
    f_pred: "RelationalField | None" = None,
    kappa_max: float = 1.0,
    N_eval: int = 16,
) -> float:
    """ΔS_shadow — total shadow correction from all PRODUCTIVE contradictions.

    For each productive contradiction C:
        ΔS_shadow += ∮ F^-_C(κ,κ̄) · (1−κ/κ_max)² · dκ̄

    The integral over κ̄ is evaluated by numerical quadrature on the circle
    |κ̄| = κ_max (contour of the conjugate variable).

    Magnitude estimate for one contradiction with σ=0.7, κ_C=0.5:
        |ΔS_shadow| ≈ 2π · 0.49 ≈ 3.08  (§12.3.4)

    Reference: §12.3.3
    """
    contradictions = _extract_productive_contradictions(field, f_pred)
    if not contradictions:
        return 0.0

    total = 0.0
    for ctr in contradictions:
        # Quadrature on the conjugate circle
        delta_s_ctr = _contour_shadow_integral(ctr, kappa_max, N_eval)
        total += delta_s_ctr
    return total


def complete_tasf(
    field: "RelationalField",
    f_pred: "RelationalField | None",
    goal_manifold=None,
    kappa_max: float = 1.0,
) -> CompleteResult:
    """Complete TASF: ΔS_mock + ΔS_shadow.

    ΔS_complete = ΔS_mock (contour integral, holomorphic)
                + ΔS_shadow (Eichler integral, non-holomorphic)
    """
    from toris.engine.tasf import TASF
    if f_pred is None:
        # Use a copy of the field as prediction (no surprise = mock only)
        f_pred = field

    tasf = TASF(N_quadrature=32, kappa_max=kappa_max)
    report = tasf.compute(field, f_pred, goal_manifold)
    delta_s_mock = report.delta_S_analytic

    delta_s_shadow = shadow_correction(field, f_pred, kappa_max)

    delta_s_complete = delta_s_mock + delta_s_shadow

    denom = abs(delta_s_complete)
    shadow_frac = abs(delta_s_shadow) / denom if denom > 1e-12 else 0.0

    return CompleteResult(
        delta_S_mock=delta_s_mock,
        delta_S_shadow=delta_s_shadow,
        delta_S_complete=delta_s_complete,
        shadow_fraction=shadow_frac,
    )


# ------------------------------------------------------------------ #
# Internal helpers                                                    #
# ------------------------------------------------------------------ #

def _extract_productive_contradictions(
    field: "RelationalField",
    f_pred: "RelationalField | None",
) -> List[ShadowContradiction]:
    """Extract PRODUCTIVE contradiction pairs from field and optional pred."""
    from toris.primitives.relation_types import d_type as _d_type

    result: List[ShadowContradiction] = []

    if f_pred is None:
        return result

    obs_index = field.relator_index()
    pred_index = f_pred.relator_index()

    for edge, r_obs in obs_index.items():
        r_pred = pred_index.get(edge)
        if r_pred is None:
            continue
        type_dist = _d_type(r_pred.tau, r_obs.tau)
        if type_dist < 0.65:
            continue  # not a genuine contradiction
        tau_a = TAU_INDEX.get(r_pred.tau, 1)
        tau_b = TAU_INDEX.get(r_obs.tau, 1)
        kappa_C = (r_pred.kappa + r_obs.kappa) / 2.0
        result.append(ShadowContradiction(
            relator_a_sigma=r_pred.sigma,
            relator_b_sigma=r_obs.sigma,
            kappa_C=kappa_C,
            tau_diff=abs(tau_a - tau_b),
            residue_strength=r_pred.sigma * r_obs.sigma * type_dist,
        ))

    return result


def _contour_shadow_integral(
    ctr: ShadowContradiction,
    kappa_max: float,
    N_eval: int,
) -> float:
    """Numerically evaluate the shadow contour integral for one contradiction.

    ΔS_shadow_C = ∮ F^-_C(κ,κ̄) · (1−κ/κ_max)² dκ̄
    where the conjugate variable κ̄ runs on the circle |κ̄| = kappa_max.

    This is non-holomorphic — it captures the persistent tension from the
    productive contradiction (§12.3.5).
    """
    total: complex = 0.0 + 0j
    for j in range(N_eval):
        # Conjugate quadrature point: κ̄_j = κ_max · exp(−2πi·j/N)
        kappa_bar = kappa_max * cmath.exp(-2j * math.pi * j / N_eval)
        # κ = conjugate of κ̄_j
        kappa = kappa_bar.conjugate()
        # double-zero suppression
        weight = (1.0 - kappa / kappa_max) ** 2
        # shadow density at this point
        f_minus = shadow_density(ctr, kappa, kappa_max)
        total += weight * f_minus

    # Prefactor: (2πi / N) for conjugate quadrature
    prefactor = (2j * math.pi / N_eval)
    return (prefactor * total).real
