"""Rogers-Ramanujan Identities as Relational Partition Function (TORIS §11.4).

First Rogers-Ramanujan identity:
    1 + Σ_{n≥1} q^(n²) / ((1-q)(1-q²)…(1-qⁿ))
        = Π_{n≥1} 1/((1-q^(5n-4))(1-q^(5n-1)))

Left side: partitions where consecutive parts differ by ≥ 2 (gap condition).
Right side: parts ≡ ±1 (mod 5).

In TORIS, valid relator configurations are those where no two CONTRADICTS-
related relators are simultaneously active (hard-exclusion, like hard hexagons).
For a chain CONTRA structure, Z_F(q) = RR product formula.

Field entropy H(F) = -d/dq [log Z_F(q)] |_{q=1/e}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import numpy as np

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

from toris.primitives.relation_types import RelationType


# ─── CONTRA graph structure ───────────────────────────────────────────────────

def _build_contra_graph(field: "RelationalField") -> Dict[str, Set[str]]:
    """Return adjacency dict of CONTRADICTS edges (concept-level, undirected)."""
    graph: Dict[str, Set[str]] = {}
    for r in field.relators():
        if r.tau == RelationType.CONTRADICTS:
            graph.setdefault(r.src_id, set()).add(r.tgt_id)
            graph.setdefault(r.tgt_id, set()).add(r.src_id)
    return graph


def contra_chain_structure(field: "RelationalField") -> bool:
    """Return True if the CONTRA graph is a simple path (linear chain).

    Conditions:
    1. All nodes have degree ≤ 2.
    2. At most 2 nodes have degree 1 (the endpoints).
    3. The graph is connected (or empty).
    """
    g = _build_contra_graph(field)
    if not g:
        return True  # empty CONTRA graph trivially qualifies

    degrees = {n: len(nbrs) for n, nbrs in g.items()}
    if any(d > 2 for d in degrees.values()):
        return False
    endpoints = sum(1 for d in degrees.values() if d == 1)
    if endpoints > 2:
        return False
    # Check connectivity
    start = next(iter(g))
    visited: Set[str] = set()
    queue = [start]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(g.get(node, []))
    return len(visited) == len(g)


# ─── Partition function ───────────────────────────────────────────────────────

def partition_function_rr(field, q: float, n_terms: int = 30) -> float:
    """Z_F(q) via first Rogers-Ramanujan infinite product formula.

    Z_F(q) = Π_{n≥1} 1/((1-q^(5n-4))(1-q^(5n-1)))

    Truncated at n_terms. Requires |q| < 1 for convergence.
    Returns 1.0 if q ≥ 1 (diverges at q=1).
    """
    if q >= 1.0:
        return float("inf")
    if q <= 0.0:
        return 1.0

    Z = 1.0
    for n in range(1, n_terms + 1):
        a = 1.0 - q ** (5 * n - 4)
        b = 1.0 - q ** (5 * n - 1)
        if abs(a) < 1e-12 or abs(b) < 1e-12:
            break
        Z /= (a * b)
    return Z


def partition_function_exact(
    field: "RelationalField",
    q: float,
    max_config_size: int = 15,
) -> float:
    """Z_F(q) by explicit enumeration of valid relator subsets.

    Valid subset = no two relators whose concepts form a CONTRA edge.
    Z_F(q) = Σ_{valid C ⊆ relators} q^|C|.

    For large fields this is exponential; limit with max_config_size.
    """
    relators = list(field.relators())
    contra_g = _build_contra_graph(field)

    def is_valid(subset: Tuple) -> bool:
        for r1, r2 in combinations(subset, 2):
            # Direct contradiction: src1→tgt2 or src2→tgt1 in CONTRA graph
            if r2.src_id in contra_g.get(r1.src_id, set()):
                return False
            if r2.tgt_id in contra_g.get(r1.tgt_id, set()):
                return False
        return True

    Z = 1.0  # empty config
    for size in range(1, min(max_config_size, len(relators)) + 1):
        for subset in combinations(relators, size):
            if is_valid(subset):
                Z += q ** size
    return Z


# ─── Field entropy ────────────────────────────────────────────────────────────

def field_entropy(field: "RelationalField", q: float = 1.0 / math.e, n_terms: int = 30) -> float:
    """H(F) = -d/dq [log Z_F(q)] at q = 1/e — field informational richness.

    From the RR product formula:
    H(F) = Σ_{n≥1} [(4n-4)·q^(5n-4)/(1-q^(5n-4)) + (4n-1)·q^(5n-1)/(1-q^(5n-1))]

    Higher H(F) = more valid configurations = higher reasoning flexibility.
    """
    H = 0.0
    for n in range(1, n_terms + 1):
        exp_a = 5 * n - 4
        exp_b = 5 * n - 1
        qa = q ** exp_a
        qb = q ** exp_b
        denom_a = 1.0 - qa
        denom_b = 1.0 - qb
        if abs(denom_a) > 1e-12:
            H += (4 * n - 4) * qa / denom_a
        if abs(denom_b) > 1e-12:
            H += (4 * n - 1) * qb / denom_b
    return H


# ─── Critical points ─────────────────────────────────────────────────────────

@dataclass
class CriticalPoint:
    kappa: float
    Z_value: float
    fractional_part: float   # |Z - round(Z)|; smaller = more critical


def critical_points(
    field: "RelationalField",
    kappa_range: Tuple[float, float] = (0.05, 0.95),
    steps: int = 200,
    threshold: float = 1e-4,
    n_terms: int = 20,
) -> List[CriticalPoint]:
    """Find κ ∈ kappa_range where Z_F(κ) is nearly an integer (TORIS §11.5.2).

    Uses the RR product formula with q = κ.
    """
    results: List[CriticalPoint] = []
    for kappa in np.linspace(kappa_range[0], kappa_range[1], steps):
        kappa_f = float(kappa)
        Z = partition_function_rr(field, kappa_f, n_terms)
        frac = abs(Z - round(Z))
        if frac < threshold:
            results.append(CriticalPoint(kappa=kappa_f, Z_value=Z, fractional_part=frac))
    return results
