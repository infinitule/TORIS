"""Relational Suppression Theorem — partition congruences as structural zeros
in relational field surprise (TORIS §11.2).

Ramanujan's partition congruences:
    p(5m+4)  ≡ 0 (mod 5)
    p(7m+5)  ≡ 0 (mod 7)
    p(11m+6) ≡ 0 (mod 11)

TORIS Suppression Theorem (conjectured):
    For fields with p-modular structure, S_{pm+r₀}(F) ≡ 0 (mod p).

Residue offsets: p=5 → r₀=4; p=7 → r₀=5; p=11 → r₀=6.

At suppressed depths, entire classes of surprise contributions cancel
exactly — TORIS can skip these depths without approximation error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from toris.field.relational_field import RelationalField

# Ramanujan residue offsets: prime → r₀
RAMANUJAN_RESIDUES: Dict[int, int] = {5: 4, 7: 5, 11: 6}


# ─── Modular structure detection ─────────────────────────────────────────────

def is_modular_field(field: "RelationalField", p: int) -> bool:
    """Return True if field relators have p-modular strength structure.

    A field is p-modular if, for each relation type, all relator strengths
    at the same depth are congruent modulo p (when discretised to integers
    in [0, p)).  In practice we check whether the *rounded* strengths
    show near-zero variance modulo p within each type-group.

    We use a relaxed criterion: the coefficient of variation of
    (σ · p) mod p across each type group is < 0.5.
    """
    from collections import defaultdict
    from toris.primitives.relation_types import RelationType

    type_groups: Dict[RelationType, List[float]] = defaultdict(list)
    for r in field.relators():
        type_groups[r.tau].append(r.sigma)

    if not type_groups:
        return False

    modular_types = 0
    for tau, sigmas in type_groups.items():
        if len(sigmas) < 2:
            modular_types += 1
            continue
        vals = [(s * p) % p for s in sigmas]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(variance)
        cv = std / (mean + 1e-9)
        if cv < 0.5:
            modular_types += 1

    return modular_types / len(type_groups) >= 0.6


def suppressed_depth(d: int, p_list: Optional[List[int]] = None) -> bool:
    """Return True if depth d is suppressed by some Ramanujan congruence.

    Does NOT check modular structure of the field — use this for the
    arithmetic check alone; pair with is_modular_field for full gating.
    """
    if p_list is None:
        p_list = list(RAMANUJAN_RESIDUES.keys())
    for p in p_list:
        r0 = RAMANUJAN_RESIDUES.get(p)
        if r0 is not None and d % p == r0:
            return True
    return False


# ─── Surprise at depth ────────────────────────────────────────────────────────

def surprise_at_depth(field: "RelationalField", d: int) -> float:
    """S_d(F) = Σ_{R: depth(R)=d} ε(R)  (TORIS §11.2.2).

    Depth is approximated by hop count from any source concept:
    depth(R) = min hop distance from any root-level concept to R.src.

    For simplicity we use the BFS layer of each relator's src_id
    in the undirected graph, then assign depth = BFS layer + 1.
    """
    import networkx as nx

    relators = list(field.relators())
    if not relators:
        return 0.0

    # Build undirected BFS from all sources with no incoming edges
    g = nx.DiGraph()
    for r in relators:
        g.add_edge(r.src_id, r.tgt_id)

    roots = [n for n in g.nodes if g.in_degree(n) == 0]
    if not roots:
        roots = list(g.nodes)[:1]

    depth_map: Dict[str, int] = {}
    for root in roots:
        for layer, nodes in enumerate(nx.bfs_layers(g, root)):
            for node in nodes:
                if node not in depth_map or layer < depth_map[node]:
                    depth_map[node] = layer

    total = 0.0
    for r in relators:
        r_depth = depth_map.get(r.src_id, 0) + 1
        if r_depth == d:
            total += r.epsilon
    return total


# ─── Suppression report ───────────────────────────────────────────────────────

@dataclass
class SuppressionEntry:
    depth: int
    s_d: float
    expected_suppressed: bool
    prime: Optional[int]
    residue: Optional[int]
    s_d_mod_p: Optional[int]
    actually_suppressed: bool   # s_d rounds to 0 mod p


@dataclass
class SuppressionReport:
    entries: List[SuppressionEntry]
    n_suppressed_correct: int    # predicted suppressed AND actually suppressed
    n_suppressed_wrong: int      # predicted but NOT actually suppressed
    suppression_accuracy: float


def suppression_report(
    field: "RelationalField",
    max_depth: int = 30,
    discretise_scale: float = 100.0,
) -> SuppressionReport:
    """Compute S_d for d=1..max_depth and check Ramanujan suppression.

    Args:
        field:             The relational field to analyse.
        max_depth:         Maximum depth to evaluate.
        discretise_scale:  S_d is scaled by this before modular check.
                           Default 100 maps [0,1] surprises to integers.
    """
    entries: List[SuppressionEntry] = []
    n_correct = 0
    n_wrong = 0

    for d in range(1, max_depth + 1):
        s_d = surprise_at_depth(field, d)
        exp_sup = suppressed_depth(d)

        prime: Optional[int] = None
        residue: Optional[int] = None
        s_d_mod_p: Optional[int] = None
        act_sup = False

        for p, r0 in RAMANUJAN_RESIDUES.items():
            if d % p == r0:
                prime = p
                residue = r0
                s_d_int = round(s_d * discretise_scale)
                s_d_mod_p = s_d_int % p
                act_sup = s_d_mod_p == 0
                break

        entries.append(SuppressionEntry(
            depth=d,
            s_d=s_d,
            expected_suppressed=exp_sup,
            prime=prime,
            residue=residue,
            s_d_mod_p=s_d_mod_p,
            actually_suppressed=act_sup,
        ))

        if exp_sup:
            if act_sup:
                n_correct += 1
            else:
                n_wrong += 1

    total_pred = n_correct + n_wrong
    accuracy = n_correct / total_pred if total_pred else 1.0
    return SuppressionReport(
        entries=entries,
        n_suppressed_correct=n_correct,
        n_suppressed_wrong=n_wrong,
        suppression_accuracy=accuracy,
    )


def verify_suppression(field: "RelationalField", d: int, p: int) -> Tuple[float, int, bool]:
    """Check S_d mod p for a single depth.

    Returns (S_d, S_d_int mod p, is_zero_mod_p).
    """
    s_d = surprise_at_depth(field, d)
    s_int = round(s_d * 100)
    return s_d, s_int % p, (s_int % p == 0)
