"""The topological surprise metric ΔS (MATH_SPEC §3).

Surprise in TORIS is NOT Euclidean or cosine distance. It is *topological
deviation* between a predicted field and an observed field, decomposed into
three components (MATH_SPEC §3.2):

    ΔS(F_pred, F_obs) = α·ΔS_struct + β·ΔS_type + γ·ΔS_strength

with α > β > γ (structural surprises matter most, strength least). Per-relator
surprise ε(R) (MATH_SPEC §3.3) gates propagation: only relators with ε > θ_ε
propagate, so confirmed predictions are suppressed at source and consume no
compute. This is predictive coding made structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from toris.constants import ALPHA, BETA, GAMMA, THETA_EPSILON
from toris.primitives.relation_types import d_type
from toris.primitives.relator import Relator
from toris.engine import tfsa

# A predicted/observed relation indexed by its directed (src, tgt) edge.
EdgeIndex = Dict[Tuple[str, str], Relator]


@dataclass
class RelatorSurprise:
    """The surprise ε(R) of a single observed relator, decomposed (§3.3)."""

    relator: Relator
    predicted: bool  # was a relation predicted on this edge?
    eps_struct: float  # 1 if unpredicted, else 0
    eps_type: float  # D_type(τ_pred, τ_obs), else 0
    eps_strength: float  # (σ_pred − σ_obs)², else 0
    epsilon: float  # α·struct + β·type + γ·strength

    def propagates(self, theta_epsilon: float = THETA_EPSILON) -> bool:
        """Propagation gate: True iff ε(R) > θ_ε (MATH_SPEC §3.3)."""
        return self.epsilon > theta_epsilon


@dataclass
class SurpriseReport:
    """Aggregate ΔS plus per-relator surprise for an observed field."""

    delta_s: float
    delta_s_struct: float
    delta_s_type: float
    delta_s_strength: float
    per_relator: Dict[int, RelatorSurprise] = field(default_factory=dict)
    theta_epsilon: float = THETA_EPSILON

    def propagating(self) -> List[Relator]:
        """Relators whose surprise exceeds θ_ε — the only ones that propagate."""
        return [
            rs.relator
            for rs in self.per_relator.values()
            if rs.propagates(self.theta_epsilon)
        ]

    def suppressed(self) -> List[Relator]:
        """Confirmed-prediction relators (ε ≤ θ_ε) — suppressed, no compute."""
        return [
            rs.relator
            for rs in self.per_relator.values()
            if not rs.propagates(self.theta_epsilon)
        ]

    def num_processing_events(self) -> int:
        """Count of relators that propagate (one processing event each)."""
        return len(self.propagating())


class SurpriseMetric:
    """Computes ΔS and per-relator ε between a predicted and observed field."""

    def __init__(
        self,
        alpha: float = ALPHA,
        beta: float = BETA,
        gamma: float = GAMMA,
        theta_epsilon: float = THETA_EPSILON,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta_epsilon = theta_epsilon

    # -- ΔS components (MATH_SPEC §3.2) ------------------------------------
    @staticmethod
    def structural_delta(pred: EdgeIndex, obs: EdgeIndex) -> float:
        """ΔS_struct = (|E_obs∖E_pred| + |E_pred∖E_obs|) / (|E_pred| + 1)."""
        e_pred, e_obs = set(pred), set(obs)
        sym_diff = len(e_obs - e_pred) + len(e_pred - e_obs)
        return sym_diff / (len(e_pred) + 1)

    @staticmethod
    def type_delta(pred: EdgeIndex, obs: EdgeIndex) -> float:
        """ΔS_type = (1/(|E_match|+1)) · Σ D_type(τ_pred(e), τ_obs(e))."""
        matched = set(pred) & set(obs)
        total = sum(d_type(pred[e].tau, obs[e].tau) for e in matched)
        return total / (len(matched) + 1)

    @staticmethod
    def strength_delta(pred: EdgeIndex, obs: EdgeIndex) -> float:
        """ΔS_strength = (1/(|E_match|+1)) · Σ (σ_pred(e) − σ_obs(e))²."""
        matched = set(pred) & set(obs)
        total = sum((pred[e].sigma - obs[e].sigma) ** 2 for e in matched)
        return total / (len(matched) + 1)

    # -- aggregate ΔS -------------------------------------------------------
    def topological_surprise(self, pred: EdgeIndex, obs: EdgeIndex) -> float:
        """ΔS = α·ΔS_struct + β·ΔS_type + γ·ΔS_strength (MATH_SPEC §3.2)."""
        return (
            self.alpha * self.structural_delta(pred, obs)
            + self.beta * self.type_delta(pred, obs)
            + self.gamma * self.strength_delta(pred, obs)
        )

    # -- per-relator surprise ε(R) (MATH_SPEC §3.3) ------------------------
    def relator_surprise(self, relator: Relator, pred: EdgeIndex) -> RelatorSurprise:
        """Surprise ε(R) for one observed relator against the prediction index."""
        edge = (relator.src_id, relator.tgt_id)
        predicted_r = pred.get(edge)
        if predicted_r is None:
            eps_struct, eps_type, eps_strength = 1.0, 0.0, 0.0
        else:
            eps_struct = 0.0
            eps_type = d_type(predicted_r.tau, relator.tau)
            eps_strength = (predicted_r.sigma - relator.sigma) ** 2
        epsilon = (
            self.alpha * eps_struct + self.beta * eps_type + self.gamma * eps_strength
        )
        return RelatorSurprise(
            relator=relator,
            predicted=predicted_r is not None,
            eps_struct=eps_struct,
            eps_type=eps_type,
            eps_strength=eps_strength,
            epsilon=epsilon,
        )

    def report(self, f_pred, f_obs) -> SurpriseReport:
        """Full surprise report comparing predicted field vs observed field.

        Accepts RelationalField instances (anything exposing ``relator_index``
        and ``relators``). Aggregate ΔS uses one representative relation per
        edge; per-relator ε is computed for *every* observed relator, including
        parallel (contradictory) ones, per MATH_SPEC §3.3 ("for each Relator R
        in F_obs").
        """
        pred = f_pred.relator_index()
        obs = f_obs.relator_index()
        per_relator = {r.rid: self.relator_surprise(r, pred) for r in f_obs.relators()}
        return SurpriseReport(
            delta_s=self.topological_surprise(pred, obs),
            delta_s_struct=self.structural_delta(pred, obs),
            delta_s_type=self.type_delta(pred, obs),
            delta_s_strength=self.strength_delta(pred, obs),
            per_relator=per_relator,
            theta_epsilon=self.theta_epsilon,
        )

    def tfsa_screen(self, f_obs) -> Tuple[List[Relator], List[Relator]]:
        """Fast surprise screen (§9.5 step 1).
        Partition all observed relators into (HIGH_SURPRISE, SUPPRESSED) in O(|E|).
        """
        return tfsa.screen_relators(f_obs.relators())

    def combined_pipeline(self, f_pred, f_obs) -> SurpriseReport:
        """The O(|E| log |E|) surprise pipeline (§9.5).

        1. TFSA screen for high-surprise candidates.
        2. Full topological ΔS computed only for candidates.
        3. Low-surprise relators are marked with ε=0.
        """
        high, suppressed = self.tfsa_screen(f_obs)

        # Compute full surprise only for screened relators
        pred_index = f_pred.relator_index()
        per_relator: Dict[int, RelatorSurprise] = {}

        for r in high:
            per_relator[r.rid] = self.relator_surprise(r, pred_index)

        for r in suppressed:
            # Assign a nominal low surprise to suppressed relators
            # This allows them to be tracked but not to propagate.
            per_relator[r.rid] = RelatorSurprise(
                relator=r,
                predicted=True, # assumed predicted to avoid structural surprise
                eps_struct=0.0,
                eps_type=0.0,
                eps_strength=0.0,
                epsilon=0.0
            )

        # Compute aggregate ΔS only on the high-surprise subset
        # We create a temporary observed index for the subset to reuse existing methods
        obs_subset = { (r.src_id, r.tgt_id): r for r in high }
        pred_subset = { k: v for k, v in pred_index.items() if k in obs_subset }

        return SurpriseReport(
            delta_s=self.topological_surprise(pred_subset, obs_subset),
            delta_s_struct=self.structural_delta(pred_subset, obs_subset),
            delta_s_type=self.type_delta(pred_subset, obs_subset),
            delta_s_strength=self.strength_delta(pred_subset, obs_subset),
            per_relator=per_relator,
            theta_epsilon=self.theta_epsilon,
        )
