"""Layers 1–2 + 6–7 + 8 + 9 — surprise computation, predictive engine, FSD, ASF, Ramanujan, Exact Surprise."""

from toris.engine.predictive import PredictiveEngine
from toris.engine.surprise import (
    RelatorSurprise,
    SurpriseMetric,
    SurpriseReport,
)
from toris.engine.fsd_pipeline import FastSurpriseDynamics, FSDReport

# Layer 7 — Analytic Surprise Functional (ASF)
from toris.engine.complex_salience import ComplexSalienceField
from toris.engine.tasf import TASF, TASFReport
from toris.engine.relational_ope import RelationalOPE, OPEExpansion
from toris.engine.michel_parameters import (
    MichelParameters,
    compute as compute_michel,
    michel_alert,
    standard_values as michel_standard,
)
from toris.engine.running_coupling import SurpriseCoupling

# Layer 9 — Exact Surprise (Rademacher, Eisenstein, Maass, Unified)
from toris.engine.rademacher import (
    rademacher_surprise, certified_surprise, RademacherResult, bessel_I_3_2
)
from toris.engine.eisenstein import (
    P_series, Q_series, R_series, eisenstein_weights, tau_function
)
from toris.engine.maass_completion import (
    shadow_correction, complete_tasf, CompleteResult, ShadowContradiction
)
from toris.engine.complete_surprise import UnifiedSurprise, UnifiedResult

# Layer 8 — Ramanujan Extension
from toris.engine.circle_method import (
    circle_method_surprise, saddle_point, kloosterman_correction,
    saddle_surprise_profile, CircleMethodResult,
)
from toris.engine.suppression import (
    is_modular_field, suppressed_depth, suppression_report,
    verify_suppression, SuppressionReport,
)
from toris.engine.ramanujan_goal import (
    goal_coherence, near_integer_check, full_warp, ramanujan_3term,
    auto_warp, pi_ramanujan,
)
from toris.engine.rogers_ramanujan import (
    contra_chain_structure, partition_function_rr, partition_function_exact,
    field_entropy, critical_points as rr_critical_points,
)
from toris.engine.ramanujan_critical import (
    RamanujanCritical, find_critical_points, is_at_critical, critical_report,
)

__all__ = [
    # Layers 1-2
    "PredictiveEngine",
    "SurpriseMetric",
    "SurpriseReport",
    "RelatorSurprise",
    # Layer 6
    "FastSurpriseDynamics",
    "FSDReport",
    # Layer 7
    "ComplexSalienceField",
    "TASF",
    "TASFReport",
    "RelationalOPE",
    "OPEExpansion",
    "MichelParameters",
    "compute_michel",
    "michel_alert",
    "michel_standard",
    "SurpriseCoupling",
    # Layer 9
    "rademacher_surprise",
    "certified_surprise",
    "RademacherResult",
    "bessel_I_3_2",
    "P_series",
    "Q_series",
    "R_series",
    "eisenstein_weights",
    "tau_function",
    "shadow_correction",
    "complete_tasf",
    "CompleteResult",
    "ShadowContradiction",
    "UnifiedSurprise",
    "UnifiedResult",
    # Layer 8 — Ramanujan
    "circle_method_surprise",
    "saddle_point",
    "kloosterman_correction",
    "saddle_surprise_profile",
    "CircleMethodResult",
    "is_modular_field",
    "suppressed_depth",
    "suppression_report",
    "verify_suppression",
    "SuppressionReport",
    "goal_coherence",
    "near_integer_check",
    "full_warp",
    "ramanujan_3term",
    "auto_warp",
    "pi_ramanujan",
    "contra_chain_structure",
    "partition_function_rr",
    "partition_function_exact",
    "field_entropy",
    "rr_critical_points",
    "RamanujanCritical",
    "find_critical_points",
    "is_at_critical",
    "critical_report",
]
