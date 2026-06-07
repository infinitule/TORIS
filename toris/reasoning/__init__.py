"""Reasoning layer — contradiction log, chain executor, and inference loop."""

from toris.reasoning.chain import ChainResult, ReasoningChain
from toris.reasoning.contradiction import (
    ContradictionEntry,
    ContradictionLog,
    ResolutionStatus,
)
from toris.reasoning.inference import InferenceLoop, StepRecord

__all__ = [
    "ContradictionLog",
    "ContradictionEntry",
    "ResolutionStatus",
    "ReasoningChain",
    "ChainResult",
    "InferenceLoop",
    "StepRecord",
]
