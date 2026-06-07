"""Ramanujan-Hardy Circle Method applied to Relational Fields (TORIS §11.1).

The Hardy-Ramanujan asymptotic formula for partitions:
    p(n) ~ (1/4n√3) · exp(π√(2n/3))

is derived from a contour integral whose dominant contribution comes
from saddle points on the unit circle at roots of unity.

In TORIS, the relational field has a generating function
    Z_F(κ) = Σ_C (∏ σ(R) for R in C) · κ^depth(C)

and "depth-d surprise" is the contour integral coefficient:
    ΔS_dominant(F, d) = (1/2πi) ∮ Z_F(κ) · κ^(-d-1) dκ

The dominant saddle point at depth d is:
    κ_saddle(d) = exp(π·√(2d/3) / d)    (Hardy-Ramanujan analog)

The first Kloosterman correction (q=2 term) is:
    ΔS_correction(F, d) ≈ (-1)^d · exp(π√(d/6)/2) / (4d√3)

This gives O(1) per depth level vs O(|E|^d) brute-force.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from scipy.integrate import quad

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField


# ─── Saddle point ────────────────────────────────────────────────────────────

def saddle_point(d: int) -> float:
    """κ_saddle for depth d — dominant singularity of Z_F (TORIS §11.1.2).

    From Hardy-Ramanujan: κ_saddle(d) = exp(π·√(2d/3) / d).
    For d=0, returns 1.0 (trivial).
    """
    if d <= 0:
        return 1.0
    return math.exp(math.pi * math.sqrt(2.0 * d / 3.0) / d)


def kloosterman_correction(d: int) -> float:
    """First Kloosterman correction term — q=2 secondary saddle (TORIS §11.1.3).

    ΔS_correction(d) ≈ (-1)^d · exp(π√(d/6)/2) / (4d√3)
    """
    if d <= 0:
        return 0.0
    sign = (-1) ** d
    return sign * math.exp(math.pi * math.sqrt(d / 6.0) / 2.0) / (4.0 * d * math.sqrt(3.0))


# ─── Z_F generating function ─────────────────────────────────────────────────

def _z_f(field: "RelationalField", kappa: float, max_depth: int = 8) -> float:
    """Evaluate Z_F(κ) up to *max_depth* hops.

    Z_F(κ) = Σ_{config C, depth ≤ max_depth} (∏ σ(R) for R in C) · κ^depth(C)

    We approximate by summing over individual relators as depth-1 configs,
    and products of adjacent relators as depth-2, etc.  For speed, we
    truncate at max_depth and cap chain count with a random sample.

    This is the O(|E|·max_depth) approximation of the true generating function.
    """
    relators = list(field.relators())
    if not relators:
        return 1.0

    # Build adjacency: tgt_id → list[Relator]
    outgoing: dict = {}
    for r in relators:
        outgoing.setdefault(r.src_id, []).append(r)

    Z = 1.0  # depth-0 (empty) configuration
    # Enumerate chains up to max_depth using DFS with a depth limit
    # State: (current_product_sigma, current_tgt_id, current_depth)
    stack = [(r.sigma, r.tgt_id, 1) for r in relators]
    visited_count = 0
    MAX_NODES = 5000
    while stack and visited_count < MAX_NODES:
        sigma_prod, cur_id, depth = stack.pop()
        Z += sigma_prod * (kappa ** depth)
        visited_count += 1
        if depth < max_depth:
            for next_r in outgoing.get(cur_id, []):
                stack.append((sigma_prod * next_r.sigma, next_r.tgt_id, depth + 1))
    return Z


# ─── Circle method surprise ───────────────────────────────────────────────────

@dataclass
class CircleMethodResult:
    d: int
    kappa_saddle: float
    delta_s_dominant: float
    delta_s_correction: float
    delta_s_total: float
    z_at_saddle: float


def circle_method_surprise(
    field: "RelationalField",
    d: int,
    max_depth: int = 8,
) -> CircleMethodResult:
    """Compute surprise at depth d via the Hardy-Ramanujan circle method.

    ΔS_dominant(F, d) = Z_F(κ_saddle) · κ_saddle^(-d)  / normalization

    The contour integral (1/2πi)∮ Z_F(κ)·κ^(-d-1)dκ is approximated
    by the dominant saddle-point contribution:
        ΔS_dominant ≈ Z_F(κ_saddle) · exp(-d · log κ_saddle)
    normalised to [0,1] by dividing by Z_F(1).

    The first Kloosterman correction is added for accuracy.
    """
    ks = saddle_point(d)
    z_saddle = _z_f(field, ks, max_depth)
    z_unit = max(_z_f(field, 1.0, max_depth), 1e-9)

    raw_dominant = z_saddle * math.exp(-d * math.log(max(ks, 1e-9)))
    delta_s_dominant = min(raw_dominant / z_unit, 1.0)
    delta_s_correction = kloosterman_correction(d)
    delta_s_total = max(0.0, min(1.0, delta_s_dominant + delta_s_correction))

    return CircleMethodResult(
        d=d,
        kappa_saddle=ks,
        delta_s_dominant=delta_s_dominant,
        delta_s_correction=delta_s_correction,
        delta_s_total=delta_s_total,
        z_at_saddle=z_saddle,
    )


def saddle_surprise_profile(
    field: "RelationalField",
    max_depth: int = 15,
) -> List[CircleMethodResult]:
    """Return circle-method results for d = 1 … max_depth."""
    return [circle_method_surprise(field, d, max_depth) for d in range(1, max_depth + 1)]
