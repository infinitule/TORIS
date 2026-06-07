"""TORIS — Topological Relational Inference System.

A ground-up computational architecture that replaces the artificial neuron,
Euclidean vector space, and backpropagation with the typed relator, the
relational field, and surprise-gradient inference.

Architecture & mathematics: Chandandeep Sharma. See the TORIS spec and
docs/MATH_SPEC.md for the governing specification.

Layers 0–4 (primitives, field, surprise/predictive engine, goal manifold, fast
plasticity) plus the reasoning loop are implemented.
"""

from toris.engine.predictive import PredictiveEngine
from toris.engine.surprise import (
    RelatorSurprise,
    SurpriseMetric,
    SurpriseReport,
)
from toris.field.relational_field import RelationalField
from toris.goal.manifold import Goal, GoalManifold
from toris.goal.subgoal import Subgoal, SubgoalStatus
from toris.plasticity.fast import (
    FastPlasticity,
    PlasticityDelta,
    TopologySnapshot,
    snapshot,
    structural_drift,
)
from toris.plasticity.medium import ConsolidationRecord, MediumPlasticity
from toris.plasticity.slow import SlowPlasticity, SlowRecord
from toris.primitives.concept_state import ConceptState
from toris.primitives.relation_types import (
    RelationType,
    can_compose_types,
    composition_rule,
    contra,
    d_type,
    is_symmetric,
    similar,
)
from toris.primitives.relator import Relator
from toris.reasoning.chain import ChainResult, ReasoningChain
from toris.reasoning.contradiction import (
    ContradictionEntry,
    ContradictionLog,
    ResolutionStatus,
)
from toris.reasoning.inference import InferenceLoop, StepRecord

__version__ = "0.1.0"

__all__ = [
    # Layer 0 — primitives
    "ConceptState",
    "Relator",
    "RelationType",
    "can_compose_types",
    "composition_rule",
    "contra",
    "d_type",
    "is_symmetric",
    "similar",
    # Layer 1 — field
    "RelationalField",
    # Layers 1–2 — surprise + predictive engine
    "SurpriseMetric",
    "SurpriseReport",
    "RelatorSurprise",
    "PredictiveEngine",
    # Layer 3 — goal manifold
    "Goal",
    "GoalManifold",
    "Subgoal",
    "SubgoalStatus",
    # Layer 4 — three-timescale plasticity + structural drift
    "FastPlasticity",
    "PlasticityDelta",
    "TopologySnapshot",
    "snapshot",
    "structural_drift",
    "MediumPlasticity",
    "ConsolidationRecord",
    "SlowPlasticity",
    "SlowRecord",
    # reasoning — contradiction log, chain executor, inference loop
    "ContradictionLog",
    "ContradictionEntry",
    "ResolutionStatus",
    "ReasoningChain",
    "ChainResult",
    "InferenceLoop",
    "StepRecord",
]
