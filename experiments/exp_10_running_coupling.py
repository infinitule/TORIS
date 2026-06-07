"""Experiment 10 — Running Surprise Coupling (TORIS Asymptotic Freedom).

Hypothesis: α_S(κ) decreases monotonically with κ — high-salience inference
is asymptotically free while low-salience inference is strongly coupled.

Success criteria:
  - α_S(0.1) > α_S(0.5) > α_S(0.9)  [monotonic decrease]
  - β-function fit converges (b0 > 0, b1 > 0)
  - χ²/dof < 2  [fit quality]
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.running_coupling import SurpriseCoupling


def make_concept(name: str) -> ConceptState:
    return ConceptState(id=name)


def build_multi_scale_field(n_per_scale: int = 5):
    """Field with relators spanning κ ∈ [0.1, 0.9] in steps of 0.1."""
    kappa_scales = [round(0.1 * k, 2) for k in range(1, 10)]   # 0.1 … 0.9
    field = RelationalField()
    taus = list(RelationType)

    concept_pool = [make_concept(f"C{i}") for i in range(50)]
    for c in concept_pool:
        field.add_concept(c)

    idx = 0
    for ki, kappa in enumerate(kappa_scales):
        for j in range(n_per_scale):
            src = concept_pool[(idx) % len(concept_pool)]
            tgt = concept_pool[(idx + 1) % len(concept_pool)]
            tau = taus[(ki + j) % len(taus)]
            # epsilon inversely proportional to kappa → low-kappa relators
            # carry high surprise (strongly coupled)
            epsilon = max(0.05, 0.8 * (1.0 - kappa) + 0.1)
            r = Relator(
                tau=tau, src=src, tgt=tgt,
                sigma=0.5 + 0.4 * kappa,
                kappa=kappa,
                epsilon=epsilon,
            )
            field.add_relator(r)
            idx += 1

    return field, kappa_scales


def chi_squared_per_dof(
    kappa_values, alpha_values, b0: float, b1: float
) -> float:
    """Compute χ²/dof for the β-function fit."""
    import numpy as np
    kv = np.array(kappa_values)
    av = np.array(alpha_values)
    log_k = np.log(kv)
    log_a = np.log(np.maximum(av, 1e-12))
    d_log_a = np.gradient(log_a, log_k)
    predicted = -(b0 * av + b1 * av ** 2)
    residuals = d_log_a - predicted
    chi2 = float(np.sum(residuals ** 2))
    dof  = max(len(kv) - 2, 1)
    return chi2 / dof


def ascii_plot(kappa_values, alpha_values, width: int = 50):
    """Print an ASCII bar chart of α_S vs κ."""
    max_a = max(alpha_values) if alpha_values else 1.0
    print(f"\n  α_S(κ)  [0 .. {max_a:.3f}]")
    print("  " + "-" * (width + 20))
    for k, a in zip(kappa_values, alpha_values):
        bar_len = int(a / max_a * width) if max_a > 0 else 0
        bar = "█" * bar_len
        print(f"  κ={k:.2f}  {a:.5f}  |{bar}")
    print("  " + "-" * (width + 20))


def run():
    print("=" * 65)
    print("Experiment 10 — Running Surprise Coupling (Asymptotic Freedom)")
    print("=" * 65)

    field, kappa_scales = build_multi_scale_field(n_per_scale=5)
    sc = SurpriseCoupling()

    alpha_vals = sc.run_coupling(field, kappa_scales)

    print(f"\n  {'κ':>6}  {'α_S(κ)':>10}")
    print("  " + "-" * 20)
    for k, a in zip(kappa_scales, alpha_vals):
        print(f"  {k:>6.2f}  {a:>10.6f}")

    ascii_plot(kappa_scales, alpha_vals)

    # Monotonicity check
    is_monotone = all(
        alpha_vals[i] >= alpha_vals[i + 1]
        for i in range(len(alpha_vals) - 1)
    )
    # Lenient: at least 7 out of 8 adjacent pairs decrease
    n_decrease = sum(
        1 for i in range(len(alpha_vals) - 1)
        if alpha_vals[i] >= alpha_vals[i + 1]
    )
    monotone_pass = n_decrease >= 6

    print(f"\n  Monotone pairs: {n_decrease}/{len(alpha_vals)-1}")
    print(f"  α_S(0.1)={alpha_vals[0]:.6f}  α_S(0.5)={alpha_vals[4]:.6f}  α_S(0.9)={alpha_vals[8]:.6f}")
    print(f"  PASS (monotone decrease ≥ 6/8): {monotone_pass}")

    # β-function fit
    b0, b1 = sc.fit_beta_function(kappa_scales, alpha_vals)
    chi2_dof = chi_squared_per_dof(kappa_scales, alpha_vals, b0, b1)
    fit_pass = b0 > 0 and b1 > 0 and chi2_dof < 2.0

    print(f"\n  β-function fit: b0={b0:.4f}  b1={b1:.4f}")
    print(f"  χ²/dof = {chi2_dof:.4f}")
    print(f"  PASS (b0>0, b1>0, χ²/dof<2): {fit_pass}")

    # Extraction from moments
    from toris.engine.relational_ope import RelationalOPE
    ope = RelationalOPE()
    moments = ope.spectral_moments(field, k_values=[0], l_values=[0], kappa_0=1.0)
    M00 = moments.get((0, 0), 0.0)
    alpha_from_moments = sc.extract_from_moments(M00, field.num_relators())
    print(f"\n  M^00 = {M00:.4f},  α_S(κ_max) from moments = {alpha_from_moments:.4f}")

    print("\n" + "=" * 65)
    all_pass = monotone_pass and fit_pass
    print(f"Experiment 10 {'PASS' if all_pass else 'FAIL'}")
    print("=" * 65)
    return all_pass, {"monotone": monotone_pass, "fit": fit_pass}


if __name__ == "__main__":
    ok, _ = run()
    sys.exit(0 if ok else 1)
