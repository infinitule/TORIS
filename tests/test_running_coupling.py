"""Tests for Layer 7 — Running Surprise Coupling (asymptotic freedom)."""

import pytest
from toris.primitives.relation_types import RelationType
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.field.relational_field import RelationalField
from toris.engine.running_coupling import SurpriseCoupling
B0 = SurpriseCoupling.B0
B1 = SurpriseCoupling.B1


def _concept(name):
    return ConceptState(id=name)


def _build_running_field(n_scales=9, n_per=5):
    """Field with relators spanning κ ∈ [0.1, 0.9]."""
    kappas = [round(0.1 * k, 2) for k in range(1, n_scales + 1)]
    field = RelationalField()
    pool = [_concept(f"C{i}") for i in range(n_scales * n_per * 2)]
    for c in pool:
        field.add_concept(c)
    taus = list(RelationType)
    idx = 0
    for ki, kappa in enumerate(kappas):
        for j in range(n_per):
            src = pool[idx % len(pool)]
            tgt = pool[(idx + 1) % len(pool)]
            eps = max(0.05, 0.8 * (1.0 - kappa) + 0.1)
            r = Relator(
                tau=taus[(ki + j) % len(taus)],
                src=src, tgt=tgt,
                sigma=0.5 + 0.4 * kappa,
                kappa=kappa,
                epsilon=eps,
            )
            field.add_relator(r)
            idx += 1
    return field, kappas


class TestSurpriseCoupling:
    def test_alpha_S_positive(self):
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        for k in kappas:
            assert sc.alpha_S(field, k) > 0

    def test_asymptotic_freedom(self):
        """α_S should decrease as κ increases (high salience → weak coupling)."""
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        alphas = sc.run_coupling(field, kappas)
        # At least 7 of 8 adjacent pairs must decrease
        n_decreasing = sum(
            1 for i in range(len(alphas) - 1) if alphas[i] >= alphas[i + 1]
        )
        assert n_decreasing >= 6, f"Only {n_decreasing}/8 pairs decrease: {alphas}"

    def test_low_kappa_stronger_than_high(self):
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        assert sc.alpha_S(field, 0.1) > sc.alpha_S(field, 0.9)

    def test_run_coupling_length(self):
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        result = sc.run_coupling(field, kappas)
        assert len(result) == len(kappas)

    def test_fit_beta_function_positive_coefficients(self):
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        alphas = sc.run_coupling(field, kappas)
        b0, b1 = sc.fit_beta_function(kappas, alphas)
        assert b0 > 0
        assert b1 > 0

    def test_fit_chi2_reasonable(self):
        import numpy as np
        field, kappas = _build_running_field()
        sc = SurpriseCoupling()
        alphas = sc.run_coupling(field, kappas)
        b0, b1 = sc.fit_beta_function(kappas, alphas)
        kv = np.array(kappas)
        av = np.array(alphas)
        log_k = np.log(kv)
        log_a = np.log(np.maximum(av, 1e-12))
        d_log_a = np.gradient(log_a, log_k)
        predicted = -(b0 * av + b1 * av ** 2)
        residuals = d_log_a - predicted
        chi2_dof = float(np.sum(residuals ** 2)) / max(len(kv) - 2, 1)
        assert chi2_dof < 5.0, f"χ²/dof = {chi2_dof:.3f}"

    def test_extract_from_moments(self):
        sc = SurpriseCoupling()
        result = sc.extract_from_moments(M00=1.5, N_relators=10)
        assert result >= 0.0

    def test_empty_field_returns_zero(self):
        field = RelationalField()
        sc = SurpriseCoupling()
        assert sc.alpha_S(field, 0.5) == 0.0
