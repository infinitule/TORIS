"""Regression guard for Experiment 06 — cross-domain consolidation."""

import importlib.util
from pathlib import Path

_EXP = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "exp_06_cross_domain_consolidation.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("exp_06", _EXP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_06_consolidation_enables_transfer():
    result = _load().run_experiment(verbose=False)
    assert result["passed"]
    assert result["monotonic"]  # §5.2 consolidation across sessions
    # the consolidated model transfers; the cold model does not reach usefulness
    assert result["consolidated_conclusion"] > result["cold_conclusion"]
    assert result["lift"] > 1.0
