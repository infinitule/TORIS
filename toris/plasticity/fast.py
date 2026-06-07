"""Fast Plasticity — topology rewriting within inference (MATH_SPEC §5.1, §5.3).

The field is not a static structure read during inference; it rewrites itself
each step:

    F^{t+1} = Φ(G^t, F^t ⊕ ΔF^t)

where the topology delta ΔF^t is driven by the surprise signal:

    ADD(R_new)     surprise revealed missing structure (ε > θ_add)
    STRENGTHEN(R)  ε > θ_strong  →  σ(R) += η_fast · ε(R)
    WEAKEN(R)      confirmed correctly N times  →  σ(R) −= η_decay
    SUPPRESS(R)    κ'(R) < θ_κ after the warp Φ

This module also implements the structural-drift measurement (§5.3) used to
verify the field is measurably different after a long reasoning chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from toris.constants import (
    CONFIRM_N,
    ETA_DECAY,
    ETA_FAST,
    THETA_ADD,
    THETA_AMPLIFY,
    THETA_EPSILON,
    THETA_KAPPA,
    THETA_STRONG,
)
from toris.engine.surprise import SurpriseReport
from toris.field.relational_field import RelationalField
from toris.primitives.relation_types import RelationType, d_type
from toris.primitives.relator import Relator


@dataclass
class PlasticityDelta:
    """The topology delta ΔF^t applied in one fast-plasticity step."""

    added: List[Relator] = field(default_factory=list)
    strengthened: List[Tuple[Relator, float, float]] = field(default_factory=list)
    weakened: List[Tuple[Relator, float, float]] = field(default_factory=list)
    suppressed: List[Relator] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"+{len(self.added)} add / {len(self.strengthened)} str / "
            f"{len(self.weakened)} weak / {len(self.suppressed)} suppress"
        )


class FastPlasticity:
    """Computes and applies ΔF^t, then warps to produce F^{t+1} (MATH_SPEC §5.1)."""

    def __init__(
        self,
        theta_add: float = THETA_ADD,
        theta_strong: float = THETA_STRONG,
        theta_epsilon: float = THETA_EPSILON,
        eta_fast: float = ETA_FAST,
        eta_decay: float = ETA_DECAY,
        confirm_n: int = CONFIRM_N,
    ) -> None:
        self.theta_add = theta_add
        self.theta_strong = theta_strong
        self.theta_epsilon = theta_epsilon
        self.eta_fast = eta_fast
        self.eta_decay = eta_decay
        self.confirm_n = confirm_n
        # consecutive-confirmation counter per relator id (for WEAKEN)
        self._confirmations: Dict[int, int] = {}

    def step(
        self,
        field: RelationalField,
        report: SurpriseReport,
        manifold,
        theta_kappa: float = THETA_KAPPA,
        theta_amplify: float = THETA_AMPLIFY,
    ) -> Tuple[RelationalField, PlasticityDelta]:
        """Apply ΔF^t to ``field`` and warp: F^{t+1} = Φ(G, F ⊕ ΔF).

        ``manifold`` warps the rewritten field (steps via Φ), which performs the
        SUPPRESS branch (κ' < θ_κ) and surfaces contradictions. Returns the new
        field F^{t+1} and the delta describing what changed.
        """
        delta = PlasticityDelta()
        report_by_rid = {rs.relator.rid: rs for rs in report.per_relator.values()}
        existing_rids = {r.rid for r in field.relators()}

        # --- ADD: surprise (ε > θ_add) revealed structure not yet in F ---
        for rs in report.per_relator.values():
            r = rs.relator
            if rs.epsilon > self.theta_add and r.rid not in existing_rids:
                field.add_relator(r)
                existing_rids.add(r.rid)
                delta.added.append(r)

        # --- STRENGTHEN / WEAKEN over the (now-augmented) field ---
        for r in field.relators():
            rs = report_by_rid.get(r.rid)
            eps = rs.epsilon if rs is not None else 0.0
            r.epsilon = eps
            if eps > self.theta_strong:
                old = r.sigma
                r.sigma = min(1.0, r.sigma + self.eta_fast * eps)
                self._confirmations[r.rid] = 0
                if r.sigma != old:
                    delta.strengthened.append((r, old, r.sigma))
            elif rs is not None and eps <= self.theta_epsilon:
                # a confirmed prediction this step
                self._confirmations[r.rid] = self._confirmations.get(r.rid, 0) + 1
                if self._confirmations[r.rid] >= self.confirm_n:
                    old = r.sigma
                    r.sigma = max(0.0, r.sigma - self.eta_decay)
                    if r.sigma != old:
                        delta.weakened.append((r, old, r.sigma))

        # --- SUPPRESS via the warp Φ: F^{t+1} = Φ(G, F ⊕ ΔF) ---
        pre_rids = {r.rid for r in field.relators()}
        warped = manifold.warp(field, theta_kappa, theta_amplify)
        post_rids = {r.rid for r in warped.relators()}
        delta.suppressed = [
            r for r in field.relators() if r.rid in (pre_rids - post_rids)
        ]
        return warped, delta


# ---------------------------------------------------------------------------
# Structural Drift Measurement (MATH_SPEC §5.3)
# ---------------------------------------------------------------------------


@dataclass
class TopologySnapshot:
    """A frozen view of a field's topology: edges, types, strengths."""

    edges: frozenset
    type_of: Dict[Tuple[str, str], RelationType]
    sigma_of: Dict[Tuple[str, str], float]

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def snapshot(field: RelationalField) -> TopologySnapshot:
    """Capture a field's topology for later drift comparison (§5.3).

    Uses one representative relator per directed (src, tgt) edge (the strongest,
    via ``relator_index``) to define τ(e) and σ(e), matching the edge identity
    used by the surprise metric.
    """
    index = field.relator_index()
    edges = frozenset(index.keys())
    type_of = {e: r.tau for e, r in index.items()}
    sigma_of = {e: r.sigma for e, r in index.items()}
    return TopologySnapshot(edges=edges, type_of=type_of, sigma_of=sigma_of)


def structural_drift(f0: TopologySnapshot, ft: TopologySnapshot) -> Dict[str, float]:
    """d_topo(F^0, F^T) and its three components (MATH_SPEC §5.3).

    ::

        d_struct   = |E^T △ E^0| / max(|E^T|, |E^0|)
        d_type     = (1/|E^0∩E^T|) · Σ D_type(τ^0(e), τ^T(e))
        d_strength = (1/|E^0∩E^T|) · Σ (σ^0(e) − σ^T(e))²
        d_topo     = (1/3)(d_struct + d_type + d_strength)

    Matched-edge components are 0 when the fields share no edges (no overlap to
    measure type/strength change over).
    """
    e0, et = f0.edges, ft.edges
    denom = max(len(e0), len(et))
    d_struct = len(e0 ^ et) / denom if denom else 0.0

    matched = e0 & et
    if matched:
        d_type = sum(d_type_(f0, ft, e) for e in matched) / len(matched)
        d_strength = sum((f0.sigma_of[e] - ft.sigma_of[e]) ** 2 for e in matched) / len(
            matched
        )
    else:
        d_type = 0.0
        d_strength = 0.0

    d_topo = (d_struct + d_type + d_strength) / 3.0
    return {
        "d_struct": d_struct,
        "d_type": d_type,
        "d_strength": d_strength,
        "d_topo": d_topo,
    }


def d_type_(f0: TopologySnapshot, ft: TopologySnapshot, edge) -> float:
    return d_type(f0.type_of[edge], ft.type_of[edge])
