"""Regression guards for Experiments 01 (retention) and 02 (goal warp)."""

import importlib.util
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _EXP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_01_contradiction_retained():
    exp = _load("exp_01_contradiction_retention")
    result = exp.run_experiment(verbose=False)
    assert result["passed"]
    assert result["retained_every_step"]
    # two distinct typed relations survive where softmax would keep one
    assert result["final_distinct_relators"] == 2
    assert result["softmax_baseline_would_keep"] == 1
    assert result["final_status"] == "PRODUCTIVE"


def test_experiment_02_goal_warp_changes_topology():
    exp = _load("exp_02_goal_warp_sensitivity")
    result = exp.run_experiment(verbose=False)
    assert result["passed"]
    assert result["edges_differ"]
    assert result["contra_differ"]
    # the physical contradiction is unique to the physical goal, and vice versa
    assert ("oxygen", "fire") in result["contra_phys"]
    assert ("alarm", "evacuation") in result["contra_proc"]
    assert ("oxygen", "fire") not in result["contra_proc"]
