"""Regression guard for Experiment 04 — surprise selectivity must keep passing."""

import importlib.util
from pathlib import Path

import pytest

_EXP = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "exp_04_surprise_selectivity.py"
)


def _load_experiment():
    spec = importlib.util.spec_from_file_location("exp_04", _EXP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_04_passes():
    exp = _load_experiment()
    result = exp.run_experiment(verbose=False)
    assert result["passed"]
    # only the 3 intended surprises propagate; the other 17 are suppressed
    assert result["n_processing_events"] == 3
    assert result["propagating_edges"] == result["surprising_edges"]


def test_experiment_04_concentration_above_bar():
    exp = _load_experiment()
    result = exp.run_experiment(verbose=False)
    assert result["concentration_on_surprise"] > 0.70
    # compute touches only a small fraction of the field
    assert result["fraction_of_field_processed"] == pytest.approx(3 / 20)
