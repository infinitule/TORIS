"""Topological operations on the Relational Field (the TORIS spec §2 / MATH_SPEC §4).

Provides graph-theoretic and algebraic topology measures over F:
  - Betti numbers β₀ (components), β₁ (independent cycles)
  - Topological distance d_topo between two fields (complement to plasticity drift)
  - Algebraic connectivity (Fiedler value) — how well-connected the field is
  - Cycle basis enumeration
  - Density / sparsity measures

All operations treat the RelationalField as a weighted directed graph and
derive topology from the underlying networkx representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import networkx as nx

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField


@dataclass
class TopologyMetrics:
    """Summary topology metrics for a single field snapshot."""
    n_concepts: int           # |V|
    n_relators: int           # |E|
    n_components: int         # β₀ — connected components (undirected)
    n_cycles: int             # β₁ = |E| - |V| + β₀  (cycle rank)
    density: float            # |E| / (|V|·(|V|-1)) ∈ [0,1]
    algebraic_connectivity: float  # Fiedler eigenvalue (≥0); 0 = disconnected
    avg_in_degree: float
    avg_out_degree: float
    diameter: Optional[int]   # longest shortest path; None if disconnected


def _as_digraph(field: "RelationalField") -> nx.DiGraph:
    g = nx.DiGraph()
    for r in field.relators():
        w = r.sigma * r.kappa  # edge weight = strength × salience
        if g.has_edge(r.src_id, r.tgt_id):
            g[r.src_id][r.tgt_id]["weight"] = max(
                g[r.src_id][r.tgt_id]["weight"], w
            )
        else:
            g.add_edge(r.src_id, r.tgt_id, weight=w)
    for c in field.concepts():
        if c.id not in g:
            g.add_node(c.id)
    return g


def _as_undirected(field: "RelationalField") -> nx.Graph:
    return _as_digraph(field).to_undirected()


def betti_numbers(field: "RelationalField") -> Tuple[int, int]:
    """(β₀, β₁) — connected components and independent cycle count.

    β₀ = number of weakly connected components (undirected view).
    β₁ = |E| - |V| + β₀   (first Betti number / cycle rank).
    """
    g = _as_undirected(field)
    v = g.number_of_nodes()
    e = g.number_of_edges()
    b0 = nx.number_connected_components(g)
    b1 = max(0, e - v + b0)
    return b0, b1


def algebraic_connectivity(field: "RelationalField") -> float:
    """Fiedler eigenvalue of the Laplacian (undirected, weighted).

    = 0 iff the field is disconnected.  Higher values mean stronger cohesion.
    Uses the normalized Laplacian; returns 0.0 for trivially small fields.
    """
    g = _as_undirected(field)
    if g.number_of_nodes() < 2:
        return 0.0
    try:
        return float(nx.algebraic_connectivity(g, weight="weight"))
    except Exception:
        return 0.0


def field_diameter(field: "RelationalField") -> Optional[int]:
    """Longest shortest path (hop count) in the undirected field.

    Returns None if the field is disconnected (no finite diameter).
    """
    g = _as_undirected(field)
    if not nx.is_connected(g):
        return None
    return nx.diameter(g)


def cycle_basis(field: "RelationalField") -> List[List[str]]:
    """Minimum cycle basis of the undirected field.

    Each element is an ordered list of concept ids forming a simple cycle.
    """
    g = _as_undirected(field)
    return nx.minimum_cycle_basis(g)


def compute_metrics(field: "RelationalField") -> TopologyMetrics:
    """Full topology snapshot for a field."""
    g_dir = _as_digraph(field)
    g_und = g_dir.to_undirected()

    v = g_dir.number_of_nodes()
    e = g_dir.number_of_edges()
    b0 = nx.number_connected_components(g_und)
    b1 = max(0, e - v + b0)
    density = e / (v * (v - 1)) if v > 1 else 0.0

    ac = 0.0
    try:
        if v >= 2:
            ac = float(nx.algebraic_connectivity(g_und, weight="weight"))
    except Exception:
        pass

    diam: Optional[int] = None
    if nx.is_connected(g_und) and v >= 2:
        try:
            diam = nx.diameter(g_und)
        except Exception:
            pass

    in_deg = sum(d for _, d in g_dir.in_degree()) / v if v else 0.0
    out_deg = sum(d for _, d in g_dir.out_degree()) / v if v else 0.0

    return TopologyMetrics(
        n_concepts=v,
        n_relators=e,
        n_components=b0,
        n_cycles=b1,
        density=density,
        algebraic_connectivity=ac,
        avg_in_degree=in_deg,
        avg_out_degree=out_deg,
        diameter=diam,
    )


def topological_distance(
    field_a: "RelationalField",
    field_b: "RelationalField",
) -> float:
    """Normalized topological distance d_topo(A, B) ∈ [0, 1].

    Combines three components:
      d_struct   = symmetric edge difference / max(|E_A|, |E_B|)
      d_betti    = |β₁(A) − β₁(B)| / max(β₁(A)+β₁(B), 1)
      d_connect  = |ac(A) − ac(B)| / max(max(ac(A),ac(B)), 1e-6)

    d_topo = (d_struct + d_betti + d_connect) / 3
    """
    ea = set(r.src_id + "→" + r.tgt_id for r in field_a.relators())
    eb = set(r.src_id + "→" + r.tgt_id for r in field_b.relators())
    sym_diff = len(ea.symmetric_difference(eb))
    max_e = max(len(ea), len(eb), 1)
    d_struct = sym_diff / max_e

    _, b1a = betti_numbers(field_a)
    _, b1b = betti_numbers(field_b)
    d_betti = abs(b1a - b1b) / max(b1a + b1b, 1)

    ac_a = algebraic_connectivity(field_a)
    ac_b = algebraic_connectivity(field_b)
    d_connect = abs(ac_a - ac_b) / max(max(ac_a, ac_b), 1e-6)
    d_connect = min(d_connect, 1.0)

    return (d_struct + d_betti + d_connect) / 3.0
