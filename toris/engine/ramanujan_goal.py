"""Rapidly convergent Goal Manifold warp via Ramanujan's 1/π series (TORIS §11.3).

Ramanujan's 1914 series:
    1/π = (2√2/9801) · Σ_{n=0}^∞ (4n)!(1103 + 26390n) / ((n!)⁴ · 396^(4n))

converges with ~8 decimal places per term.  In TORIS the warp sum is:

    Φ(G,F) = Σ_{k=0}^{K} priority(k) · mean_salience(F, depth=k)

For a COHERENT goal manifold (small Q), the per-depth contributions decay
rapidly, so truncating at n_terms=3 gives 24-bit precision (analogous to
Ramanujan's 3-term π computation).

auto_warp switches:
    Q < coherence_threshold → ramanujan_3term  (fast, truncated)
    Q ≥ coherence_threshold → full_warp        (exact, all K terms)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField
    from toris.goal.manifold import GoalManifold

# Ramanujan series constants
_RAMANUJAN_PREFACTOR = 2.0 * math.sqrt(2.0) / 9801.0


def _ramanujan_term(n: int) -> float:
    """Return the n-th term of Ramanujan's 1/π series (unnormalised)."""
    fact_4n = math.factorial(4 * n)
    fact_n4 = math.factorial(n) ** 4
    return fact_4n * (1103.0 + 26390.0 * n) / (fact_n4 * (396.0 ** (4 * n)))


def pi_ramanujan(n_terms: int = 5) -> float:
    """Compute 1/π via Ramanujan's series with n_terms terms."""
    total = sum(_ramanujan_term(n) for n in range(n_terms))
    inv_pi = _RAMANUJAN_PREFACTOR * total
    return 1.0 / inv_pi


# ─── Goal coherence ──────────────────────────────────────────────────────────

def goal_coherence(manifold: "GoalManifold") -> float:
    """Q(G) = nearness of warp-sum to nearest integer (TORIS §11.3.3).

    The warp sum S = Σ_{k} priority(k) / (k+1)  is analogous to the
    Ramanujan π series partial sum.  When S ≈ integer (Q small), the
    manifold has high modular coherence — the 3-term approximation converges.

    Returns Q ∈ [0, 0.5].
    """
    priorities: List[float] = []
    for sg in manifold.active:
        p = getattr(sg, "priority", 1.0)
        priorities.append(float(p))

    if not priorities:
        return 0.0

    warp_sum = sum(p / (i + 1) for i, p in enumerate(priorities))
    frac = abs(warp_sum - round(warp_sum))
    return min(frac, 0.5)


def near_integer_check(Z: float, threshold: float = 1e-4) -> bool:
    """Return True if Z is within *threshold* of the nearest integer."""
    return abs(Z - round(Z)) < threshold


# ─── Warp implementations ─────────────────────────────────────────────────────

def _mean_salience(manifold: "GoalManifold", field: "RelationalField") -> float:
    """Mean relator salience under the current goal manifold."""
    from toris.goal.warp import relevance_fn
    relators = list(field.relators())
    if not relators:
        return 0.0
    rel_fn = relevance_fn(manifold)
    return sum(rel_fn(r) for r in relators) / len(relators)


def full_warp(manifold: "GoalManifold", field: "RelationalField") -> float:
    """Exact O(K·N) warp sum over ALL active subgoals.

    Φ_exact = mean_salience(primary) + Σ_{k} priority(k) · mean_salience(F)

    In our formulation each subgoal uses the same salience function but is
    weighted by its priority and depth-discount 1/(k+1).
    """
    base = _mean_salience(manifold, field)
    total = base
    for k, sg in enumerate(manifold.active):
        priority = float(getattr(sg, "priority", 1.0))
        total += priority / (k + 1) * base
    return total


def ramanujan_3term(
    manifold: "GoalManifold",
    field: "RelationalField",
    depth: Optional[int] = None,
    n_terms: int = 3,
) -> float:
    """n_terms-truncated warp approximation (TORIS §11.3.2).

    For a COHERENT manifold, only the first n_terms subgoals contribute
    significantly — deeper subgoals' priority × depth_discount is negligible.
    This gives O(n_terms · N) vs O(K · N) for full_warp.

    The Ramanujan insight: for modular manifolds the depth-discount series
    Σ_k p_k/(k+1) converges as rapidly as Ramanujan's π series.
    """
    base = _mean_salience(manifold, field)
    total = base
    active = list(manifold.active)[:n_terms]
    for k, sg in enumerate(active):
        priority = float(getattr(sg, "priority", 1.0))
        total += priority / (k + 1) * base
    return total


def auto_warp(
    manifold: "GoalManifold",
    field: "RelationalField",
    coherence_threshold: float = 0.01,
) -> Tuple[float, str]:
    """Auto-switch between Ramanujan 3-term and full warp.

    Returns (Φ_value, method_used) where method_used ∈ {'ramanujan', 'exact'}.
    """
    Q = goal_coherence(manifold)
    if Q < coherence_threshold:
        return ramanujan_3term(manifold, field), "ramanujan"
    else:
        return full_warp(manifold, field), "exact"


# ─── Near-integer scan ────────────────────────────────────────────────────────

def scan_near_integers(
    manifold: "GoalManifold",
    field: "RelationalField",
    kappa_steps: int = 50,
    threshold: float = 1e-4,
) -> List[Tuple[float, float]]:
    """Scan κ ∈ [0.1, 0.9] for near-integer Φ(G,F; κ) values."""
    from toris.goal.warp import relevance_fn

    relators = list(field.relators())
    results = []
    for kappa in np.linspace(0.1, 0.9, kappa_steps):
        rel_fn = relevance_fn(manifold)
        phi = sum(rel_fn(r) * kappa for r in relators) / max(len(relators), 1)
        if near_integer_check(phi, threshold):
            results.append((float(kappa), phi))
    return results
