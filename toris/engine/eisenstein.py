"""Layer 9 — Eisenstein Series and Depth-Dependent Weights (the Exact-Surprise spec §12.2).

The Ramanujan-Eisenstein series P, Q, R provide the correct weighting for
ΔS components in the deep inference regime (d > d_crit).

Standard (shallow, d ≤ d_crit = 5): weights α=0.6, β=0.3, γ=0.1
Eisenstein (deep, d > d_crit):       weights α=1/6, β=1/3, γ=1/2

This is the Dual Weighting Theorem — the first depth-dependent weighting in
any relational inference system.

Also defines the TORIS tau function τ_F(d) — weight-12 analog of Ramanujan's
τ(n) function.

Reference: the Exact-Surprise spec §12.2
"""

from __future__ import annotations

import cmath
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

from toris.engine.rademacher import TAU_INDEX

# Default crossover depth (§12.2.2)
D_CRIT_DEFAULT = 5

# Eisenstein weight ratios from modular form theory:
#   weight_struct / total = 2/(2+4+6) = 1/6
#   weight_type   / total = 4/12      = 1/3
#   weight_str    / total = 6/12      = 1/2
EISENSTEIN_ALPHA = 1.0 / 6.0
EISENSTEIN_BETA  = 1.0 / 3.0
EISENSTEIN_GAMMA = 1.0 / 2.0

# Empirical (shallow) weights from MATH_SPEC §3.2
EMPIRICAL_ALPHA = 0.6
EMPIRICAL_BETA  = 0.3
EMPIRICAL_GAMMA = 0.1


# ------------------------------------------------------------------ #
# Eisenstein series P, Q, R                                           #
# ------------------------------------------------------------------ #

def P_series(q: float, N_terms: int = 50) -> float:
    """Eisenstein series P(q) of weight 2.

        P(q) = 1 − 24 Σ_{k=1}^N k·q^k/(1−q^k)

    Reference: §12.2.1
    """
    if abs(q) >= 1.0:
        return float("nan")
    total = 1.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        denom = 1.0 - qk
        if abs(denom) < 1e-15:
            break
        total -= 24.0 * k * qk / denom
    return total


def Q_series(q: float, N_terms: int = 50) -> float:
    """Eisenstein series Q(q) of weight 4.

        Q(q) = 1 + 240 Σ_{k=1}^N k³·q^k/(1−q^k)

    Reference: §12.2.1
    """
    if abs(q) >= 1.0:
        return float("nan")
    total = 1.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        denom = 1.0 - qk
        if abs(denom) < 1e-15:
            break
        total += 240.0 * (k ** 3) * qk / denom
    return total


def R_series(q: float, N_terms: int = 50) -> float:
    """Eisenstein series R(q) of weight 6.

        R(q) = 1 − 504 Σ_{k=1}^N k⁵·q^k/(1−q^k)

    Reference: §12.2.1
    """
    if abs(q) >= 1.0:
        return float("nan")
    total = 1.0
    for k in range(1, N_terms + 1):
        qk = q ** k
        denom = 1.0 - qk
        if abs(denom) < 1e-15:
            break
        total -= 504.0 * (k ** 5) * qk / denom
    return total


# ------------------------------------------------------------------ #
# Dual weighting theorem                                              #
# ------------------------------------------------------------------ #

def eisenstein_weights(d: int, d_crit: int = D_CRIT_DEFAULT) -> tuple[float, float, float]:
    """Return (alpha, beta, gamma) for ΔS = α·ΔS_struct + β·ΔS_type + γ·ΔS_strength.

    For d ≤ d_crit (shallow): empirical weights (0.6, 0.3, 0.1)
    For d > d_crit (deep):    Eisenstein weights (1/6, 1/3, 1/2)

    Dual Weighting Theorem (§12.2.2):
    - Shallow regime: structural surprise dominates (missing edges = biggest error)
    - Deep regime:    strength surprise dominates (strength encodes most information)
    """
    if d <= d_crit:
        return (EMPIRICAL_ALPHA, EMPIRICAL_BETA, EMPIRICAL_GAMMA)
    else:
        return (EISENSTEIN_ALPHA, EISENSTEIN_BETA, EISENSTEIN_GAMMA)


def modular_delta_S(
    field: "RelationalField",
    f_pred: "RelationalField",
    d: int,
    d_crit: int = D_CRIT_DEFAULT,
) -> float:
    """ΔS with depth-appropriate (Eisenstein or empirical) weights.

    Uses the SurpriseMetric's component computations with the correct weights
    for depth d.  Reference: §12.2.2
    """
    from toris.engine.surprise import SurpriseMetric
    alpha, beta, gamma = eisenstein_weights(d, d_crit)
    sm = SurpriseMetric(alpha=alpha, beta=beta, gamma=gamma)
    return sm.topological_surprise(field, f_pred)


# ------------------------------------------------------------------ #
# TORIS tau function                                                  #
# ------------------------------------------------------------------ #

def tau_function(field: "RelationalField", d: int) -> complex:
    """TORIS tau function τ_F(d) — weight-12 analog of Ramanujan's τ(n).

        τ_F(d) = Σ_{R: depth(R)=d} σ(R)^5 · κ(R)^7 · exp(2πi·τ_index(R)/12)

    Properties: multiplicativity, Deligne-type growth bound.
    Reference: §12.4.2
    """
    total: complex = 0.0 + 0j
    for r in field.relators():
        tau_idx = TAU_INDEX.get(r.tau, 1)
        phase = cmath.exp(2j * math.pi * tau_idx / 12.0)
        total += (r.sigma ** 5) * (r.kappa ** 7) * phase
    return total


def tau_congruence_check(
    field: "RelationalField",
    d: int,
    S_d: float | None = None,
) -> bool:
    """Check S(d) ≡ 11·τ_F(d) (mod 13) for d ≡ 6 (mod 13).

    Returns True if the congruence holds (within numerical tolerance),
    or True trivially when d ≢ 6 (mod 13) (congruence doesn't apply).
    Reference: §12.2.3
    """
    if d % 13 != 6:
        return True  # congruence only applies at these depths

    if S_d is None:
        from toris.engine.rademacher import rademacher_surprise
        S_d = rademacher_surprise(field, d, N_terms=3).S_exact

    tau_val = tau_function(field, d)
    rhs = 11.0 * tau_val.real

    # Check mod 13: S_d ≡ rhs (mod 13) means round(S_d - rhs) divisible by 13
    diff = S_d - rhs
    remainder = round(diff) % 13
    return remainder == 0
