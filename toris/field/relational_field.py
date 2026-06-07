"""The Relational Field — TORIS's replacement for the vector space.

The field F is a typed directed hypergraph (the TORIS spec §3.3 / MATH_SPEC):

    F = (V, E, τ_E, W, Φ)

    V    ConceptStates (nodes)
    E    Relators (directed, typed edges) — parallel edges allowed
    τ_E  type of each edge (carried by the Relator)
    W    strength of each edge (= σ of the Relator)
    Φ    the goal-driven warp operator (MATH_SPEC §4.2)

This module wraps a ``networkx.MultiDiGraph`` so that contradictory parallel
relations (e.g. CAUSAL(A→B) and NEGATES(A→B)) can coexist as live tension —
they are never collapsed. Nodes are keyed by concept id; edges are keyed by the
relator's ``rid`` so a relation stays identifiable as its σ/κ/ε mutate.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple, Union

import networkx as nx

from toris.constants import THETA_AMPLIFY, THETA_KAPPA
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator

ConceptRef = Union[ConceptState, str]
RelatorRef = Union[Relator, int]


def _concept_id(ref: ConceptRef) -> str:
    return ref.id if isinstance(ref, ConceptState) else ref


def _rid(ref: RelatorRef) -> int:
    return ref.rid if isinstance(ref, Relator) else ref


class RelationalField:
    """A typed directed hypergraph of ConceptStates linked by Relators."""

    def __init__(self) -> None:
        # MultiDiGraph: directed, parallel edges allowed (for contradictions).
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # -- concepts (V) -------------------------------------------------------
    def add_concept(self, concept: ConceptState) -> ConceptState:
        """Add a ConceptState node (idempotent on id)."""
        if not self._g.has_node(concept.id):
            self._g.add_node(concept.id, concept=concept)
        return self._g.nodes[concept.id]["concept"]

    def get_concept(self, ref: ConceptRef) -> Optional[ConceptState]:
        cid = _concept_id(ref)
        if self._g.has_node(cid):
            return self._g.nodes[cid]["concept"]
        return None

    def has_concept(self, ref: ConceptRef) -> bool:
        return self._g.has_node(_concept_id(ref))

    def concepts(self) -> List[ConceptState]:
        return [data["concept"] for _, data in self._g.nodes(data=True)]

    # -- relators (E) -------------------------------------------------------
    def add_relator(self, relator: Relator) -> Relator:
        """Add a Relator as a typed directed edge, keyed by its ``rid``.

        Endpoints are auto-added as concepts if absent. Parallel relators on the
        same (src, tgt) pair are preserved (they may contradict).
        """
        self.add_concept(relator.src)
        self.add_concept(relator.tgt)
        self._g.add_edge(
            relator.src_id, relator.tgt_id, key=relator.rid, relator=relator
        )
        return relator

    def add_relators(self, relators: Iterable[Relator]) -> "RelationalField":
        for r in relators:
            self.add_relator(r)
        return self

    def remove_relator(self, ref: RelatorRef) -> None:
        """Remove a relator by object or rid (no-op if absent)."""
        rid = _rid(ref)
        for u, v, k in list(self._g.edges(keys=True)):
            if k == rid:
                self._g.remove_edge(u, v, key=k)
                return

    def get_relator(self, ref: RelatorRef) -> Optional[Relator]:
        rid = _rid(ref)
        for _, _, data in self._g.edges(data=True):
            if data["relator"].rid == rid:
                return data["relator"]
        return None

    def relators(self) -> List[Relator]:
        return [data["relator"] for _, _, data in self._g.edges(data=True)]

    def relators_between(self, src: ConceptRef, tgt: ConceptRef) -> List[Relator]:
        """All parallel relators on the directed pair (src → tgt)."""
        u, v = _concept_id(src), _concept_id(tgt)
        if not self._g.has_edge(u, v):
            return []
        return [d["relator"] for d in self._g.get_edge_data(u, v).values()]

    # -- aggregate views ----------------------------------------------------
    def edge_set(self) -> Set[Tuple[str, str]]:
        """The set of directed (src_id, tgt_id) pairs E (untyped).

        This is the edge identity used by the surprise metric ΔS (MATH_SPEC
        §3.2), which reasons about E ⊆ V × V.
        """
        return {(u, v) for u, v in self._g.edges()}

    def num_relators(self) -> int:
        return self._g.number_of_edges()

    def num_concepts(self) -> int:
        return self._g.number_of_nodes()

    # -- relational neighborhood -------------------------------------------
    def get_neighborhood(
        self, concept: ConceptRef, depth: int = 1
    ) -> "RelationalField":
        """Return the sub-field within ``depth`` relational hops of ``concept``.

        The relational *context* of a concept includes relations pointing both
        into and out of it, so the hop expansion follows edges in both
        directions (undirected reachability). The returned field contains the
        reached concepts and every relator whose endpoints are both reached.
        """
        start = _concept_id(concept)
        sub = RelationalField()
        if not self._g.has_node(start):
            return sub

        # BFS over the undirected view to collect concept ids within `depth`.
        reached: Set[str] = {start}
        frontier: Set[str] = {start}
        undirected = self._g.to_undirected(as_view=True)
        for _ in range(max(0, depth)):
            nxt: Set[str] = set()
            for node in frontier:
                nxt.update(undirected.neighbors(node))
            nxt -= reached
            reached.update(nxt)
            frontier = nxt
            if not frontier:
                break

        for cid in reached:
            sub.add_concept(self._g.nodes[cid]["concept"])
        for _, _, data in self._g.edges(data=True):
            r = data["relator"]
            if r.src_id in reached and r.tgt_id in reached:
                sub.add_relator(r)
        return sub

    # -- the warp operator Φ (MATH_SPEC §4.2, steps 1–3) -------------------
    def warp(
        self,
        relevance: Callable[[Relator], float],
        theta_kappa: float = THETA_KAPPA,
        theta_amplify: float = THETA_AMPLIFY,
    ) -> "RelationalField":
        """Φ(G, F) → F': goal-driven topological transformation of the field.

        ``relevance`` is the combined goal-relevance multiplier for a relator
        (MATH_SPEC §4.2 step 1: ``relevance(R, G_p) · Σ_g priority(g)·relevance(R,g)``);
        the Goal Manifold (Layer 3) supplies it. This method applies the
        mechanism:

        * **Step 1** recompute salience  ``κ'(R) = κ(R) · relevance(R)`` (clamped to [0,1]).
        * **Step 2** suppress             drop relators with ``κ'(R) ≤ θ_κ``.
        * **Step 3** amplify              for ``κ'(R) > θ_amplify``: ``σ' = min(1, σ·(1+κ'))``.

        Returns a NEW field with a different topology — different edges active,
        different strengths. The original is untouched (relators are cloned with
        their ``rid`` preserved). Step 4 (surfacing goal-relevant contradictions
        to the contradiction log) belongs to the Goal Manifold layer.
        """
        warped = RelationalField()
        for c in self.concepts():
            warped.add_concept(c)
        for r in self.relators():
            kappa_new = min(1.0, max(0.0, r.kappa * float(relevance(r))))
            if kappa_new <= theta_kappa:
                continue  # suppressed: removed from the active topology
            sigma_new = r.sigma
            if kappa_new > theta_amplify:
                sigma_new = min(1.0, r.sigma * (1.0 + kappa_new))
            warped.add_relator(r.clone(sigma=sigma_new, kappa=kappa_new))
        return warped

    # -- copying ------------------------------------------------------------
    def copy(self) -> "RelationalField":
        """Deep-ish copy: relators cloned (rid preserved), concepts shared."""
        dup = RelationalField()
        for c in self.concepts():
            dup.add_concept(c)
        for r in self.relators():
            dup.add_relator(r.clone())
        return dup

    def relator_index(self) -> Dict[Tuple[str, str], Relator]:
        """Map each directed pair (src, tgt) → a single representative relator.

        Used by the surprise metric, which compares one predicted vs one
        observed relation per edge (MATH_SPEC §3). When a pair carries parallel
        relators, the strongest (max σ) is kept as the representative; holding
        the parallel tension is the contradiction log's job, not ΔS's.
        """
        index: Dict[Tuple[str, str], Relator] = {}
        for r in self.relators():
            key = (r.src_id, r.tgt_id)
            current = index.get(key)
            if current is None or r.sigma > current.sigma:
                index[key] = r
        return index

    # -- dunder -------------------------------------------------------------
    def __contains__(self, ref: RelatorRef) -> bool:
        return self.get_relator(ref) is not None

    def __len__(self) -> int:
        return self.num_relators()

    def __repr__(self) -> str:
        return (
            f"RelationalField(|V|={self.num_concepts()}, " f"|E|={self.num_relators()})"
        )
