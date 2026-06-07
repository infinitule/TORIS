"""Regression guards for Experiments 03 (sparse gen) and 05 (structural drift)."""

import importlib.util
from pathlib import Path

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _EXP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_05_structural_drift_exceeds_bar():
    exp = _load("exp_05_structural_drift")
    result = exp.run_experiment(verbose=False)
    assert result["passed"]
    assert result["drift"]["d_topo"] > 0.1
    assert result["edges_tT"] > result["edges_t0"]  # the field grew
    assert result["total_added"] > 0


def test_experiment_03_sparse_generalization_calibrated():
    exp = _load("exp_03_sparse_generalization")
    result = exp.run_experiment(verbose=False)
    assert result["passed"]
    r = result["result"]
    assert r.hops == 8 and r.n_seed == 4 and r.n_hypothetical == 4
    assert not r.broken
    # §6 conjecture holds, conclusion did not collapse to zero
    assert r.sigma_chain >= r.sigma_bound > 0.0
    # uncertainty is calibrated: far less confident than a fully-seeded chain
    assert r.sigma_chain < result["baseline_sigma_chain"]
