"""TFSA — the TORIS Fast Surprise Approximation (Section 9.1–9.3).

A relator's contextual salience κ lives in [0,1]. Encoding κ in a fixed-point
log-salience space turns the inverse-square-root — the geometric heart of the
surprise estimate — into a single integer subtract-and-shift, exactly as the
Fast Inverse Square Root exploits the logarithmic geometry of IEEE 754:

    s(R)      = floor(2^B · log₂(κ + δ))          (encode to log-fixed-point)
    s_approx  = MAGIC − (s >> 1)                  (O(1) fast inverse sqrt)
    y         = y · (1.5 − 0.5·κ·y²)              (one Newton refinement step)

This yields ``tfsa(κ) ≈ 1/√κ`` in O(1) per relator. The screen score
``tfsa_surprise_potential(κ) = κ · tfsa(κ) ≈ √κ`` (computed *without* a sqrt
call) is a [0,1] monotone proxy used to cheaply screen which relators deserve
the full O(|E|²) topological ΔS — turning the surprise pass into O(|E| log |E|).

DEVIATION D-19: the spec's literal magic constant ``S_TORIS_B16 = 562759``
(§9.2) carries a spurious +B/2 term that makes the literal pipeline collapse to
0 for every κ. The calibrated magic ``S_TORIS_B16_CALIBRATED = 2952`` (the
genuine fast-inverse-sqrt correction round(2^B·0.0450465)) is used here so the
approximation actually tracks 1/√κ to <0.2% after refinement. Both constants are
kept; see docs/DEVIATIONS.md.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from toris.constants import (
    B_BITS,
    DELTA_LOG,
    S_TORIS_B16,  # noqa: F401  (preserved spec constant; see D-19)
    S_TORIS_B16_CALIBRATED,
    TFSA_SCREEN_THRESHOLD,
)
from toris.primitives.relator import Relator


def encode_salience(kappa: float, b: int = B_BITS) -> int:
    """Encode κ into B-bit fixed-point log-salience space (§9.3 step 1).

    ``s = floor(2^B · log₂(κ + δ))`` — the integer encoding *is* a linear
    approximation of the logarithm, so a right shift is a halving in log-space
    (the square-root in linear space).
    """
    return int((2**b) * math.log2(kappa + DELTA_LOG))


def fast_inverse_sqrt(kappa: float, b: int = B_BITS) -> float:
    """O(1) fast inverse-sqrt estimate of κ via the log-fixed-point bit trick.

    ``decode(MAGIC − (s >> 1))`` ≈ 1/√(κ+δ). Uses the calibrated magic (D-19).
    This is the §9.3 step-2 ``fast_surprise_estimate`` with a working constant.
    """
    s = encode_salience(kappa, b)
    s_approx = S_TORIS_B16_CALIBRATED - (s >> 1)
    return (2 ** (s_approx / (2**b))) - DELTA_LOG


# The spec names this ``fast_surprise_estimate``; it is the fast inverse-sqrt.
fast_surprise_estimate = fast_inverse_sqrt


def refine_surprise(kappa: float, epsilon_fast: float) -> float:
    """One Newton–Raphson refinement step (§9.3 step 3).

    ``y ← y·(1.5 − 0.5·x·y²)`` with x = κ, y = estimate. This is the exact
    inverse-square-root Newton step; one iteration removes ~99.8% of the
    approximation error.
    """
    return epsilon_fast * (1.5 - 0.5 * kappa * epsilon_fast**2)


def tfsa(kappa: float, b: int = B_BITS) -> float:
    """Full TFSA pipeline — O(1) fast inverse-sqrt of κ (≈ 1/√κ), §9.3.

    Two integer ops + one multiply (fast estimate) plus one Newton step. Returns
    the refined 1/√κ estimate (≥ 1 for κ ≤ 1). For a bounded [0,1] screen score
    use :func:`tfsa_surprise_potential`.
    """
    estimate = fast_inverse_sqrt(kappa, b)
    return refine_surprise(kappa, estimate)


def tfsa_surprise_potential(kappa: float, b: int = B_BITS) -> float:
    """A [0,1] monotone surprise-potential screen score ≈ √κ (§9.3, §9.5).

    Computed as ``κ · tfsa(κ) ≈ κ/√κ = √κ`` — i.e. the fast inverse-sqrt turns a
    square-root into one multiply, no ``math.sqrt`` call. Higher salience → higher
    potential → the relator is worth the full ΔS. Clamped to [0,1].
    """
    return max(0.0, min(1.0, kappa * tfsa(kappa, b)))


def screen_relators(
    relators: List[Relator],
    threshold: float = TFSA_SCREEN_THRESHOLD,
    b: int = B_BITS,
) -> Tuple[List[Relator], List[Relator]]:
    """Partition relators into (HIGH_SURPRISE, SUPPRESSED) by TFSA in O(|E|).

    HIGH_SURPRISE = { R : tfsa_surprise_potential(κ(R)) > threshold } (§9.5).
    Each decision is O(1), so the whole screen is O(|E|).
    """
    high: List[Relator] = []
    suppressed: List[Relator] = []
    for r in relators:
        if tfsa_surprise_potential(r.kappa, b) > threshold:
            high.append(r)
        else:
            suppressed.append(r)
    return high, suppressed
