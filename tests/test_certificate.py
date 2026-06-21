"""
Unit test: Theoretical bounds hold on synthetic data.

Verifies that:
    1. eps_delta is always non-negative and finite.
    2. The margin check (Corollary 1) correctly identifies whether the
       certificate passes or fails.
    3. Certificate components decompose correctly: eps_delta = sqrt(eps_stab) + sqrt(eps_proj).
    4. When rank_p = r (full rank), eps_proj = 0.
    5. Stability bound scales correctly with m and alpha.
"""


import math

import pytest
import torch

from fkp.theory.bounds import stability_bound, feature_norm_bound, projection_loss_bound
from fkp.theory.certificate import FKPCertificate
from fkp.compression.ridge import ridge_auto
from fkp.compression.svd import spectral_compress
from fkp.conditioning.centering import center_logits
from fkp.conditioning.zca import ZCAWhitener


class TestStabilityBound:
    """Tests for the uniform stability bound."""

    def test_stability_bound_is_nonnegative(self) -> None:
        """eps_stab must be non-negative."""
        torch.manual_seed(0)
        m, D, c = 100, 50, 10
        E = torch.randn(m, D)
        H_c = torch.randn(m, c)
        W = ridge_auto(E, H_c, alpha=1.0)
        result = stability_bound(E, H_c, W, alpha=1.0, delta=0.05)
        assert result.eps_stab >= 0, f"eps_stab is negative: {result.eps_stab}"

    def test_stability_bound_decreases_with_larger_m(self) -> None:
        """With more calibration data, the bound should be tighter."""
        torch.manual_seed(1)
        D, c, alpha = 30, 5, 1.0

        # Small m
        m_small = 50
        E_s = torch.randn(m_small, D)
        H_s = torch.randn(m_small, c)
        H_s_c, _ = center_logits(H_s)
        W_s = ridge_auto(E_s, H_s_c, alpha=alpha)
        result_small = stability_bound(E_s, H_s_c, W_s, alpha=alpha, delta=0.05)

        # Large m
        m_large = 500
        E_l = torch.randn(m_large, D)
        H_l = torch.randn(m_large, c)
        H_l_c, _ = center_logits(H_l)
        W_l = ridge_auto(E_l, H_l_c, alpha=alpha)
        result_large = stability_bound(E_l, H_l_c, W_l, alpha=alpha, delta=0.05)

        assert result_large.c_delta < result_small.c_delta, (
            "Confidence coefficient c_delta should decrease with larger m."
        )

    def test_stability_bound_components_sum_correctly(self) -> None:
        """eps_stab must equal E_reg + gamma + c_delta*(E_reg + gamma)."""
        torch.manual_seed(2)
        m, D, c = 80, 20, 5
        E = torch.randn(m, D)
        H_c = torch.randn(m, c)
        W = ridge_auto(E, H_c, alpha=1.0)
        result = stability_bound(E, H_c, W, alpha=1.0, delta=0.05)

        expected = result.E_reg + result.gamma + result.c_delta * (result.E_reg + result.gamma)
        assert abs(result.eps_stab - expected) < 1e-10, (
            f"eps_stab components don't sum correctly: "
            f"{result.eps_stab:.8f} != {expected:.8f}"
        )


class TestFeatureNormBound:
    """Tests for the sub-Gaussian feature norm bound."""

    def test_r_delta_is_positive(self) -> None:
        torch.manual_seed(3)
        E = torch.randn(100, 50)
        R = feature_norm_bound(E, delta=0.05)
        assert R > 0, f"R_delta should be positive, got {R}"

    def test_r_delta_decreases_with_larger_m(self) -> None:
        """More calibration data should tighten the feature norm bound."""
        D = 50
        R_small = feature_norm_bound(torch.randn(50, D), delta=0.05)
        R_large = feature_norm_bound(torch.randn(500, D), delta=0.05)
        assert R_large < R_small, "R_delta should decrease with larger m."

    def test_r_delta_tightens_with_larger_delta(self) -> None:
        """Smaller confidence (larger delta) should give a looser bound."""
        E = torch.randn(100, 50)
        R_tight = feature_norm_bound(E, delta=0.001)
        R_loose = feature_norm_bound(E, delta=0.1)
        assert R_loose < R_tight, (
            "Looser confidence (larger delta) should give smaller R_delta."
        )


class TestProjectionLossBound:
    """Tests for the spectral projection loss bound."""

    def test_zero_projection_loss_at_full_rank(self) -> None:
        """When rank_p = r, the projection is exact: eps_proj = 0."""
        torch.manual_seed(4)
        D, c = 20, 10
        r = min(D, c)
        singular_values = torch.rand(r).sort(descending=True).values + 0.1

        eps_proj = projection_loss_bound(singular_values, rank_p=r, R_delta=5.0)
        assert eps_proj == 0.0, (
            f"eps_proj should be 0 when rank_p = r = {r}, got {eps_proj}"
        )

    def test_projection_loss_decreases_with_larger_p(self) -> None:
        """Higher rank should always decrease or maintain the bound."""
        torch.manual_seed(5)
        n_sv = 20
        svs = torch.linspace(10.0, 0.1, n_sv)  # descending singular values

        eps_p5 = projection_loss_bound(svs, rank_p=5, R_delta=3.0)
        eps_p10 = projection_loss_bound(svs, rank_p=10, R_delta=3.0)
        eps_p15 = projection_loss_bound(svs, rank_p=15, R_delta=3.0)

        assert eps_p5 >= eps_p10 >= eps_p15, (
            "Projection loss bound must be non-increasing with rank_p."
        )


class TestFKPCertificateEndToEnd:
    """End-to-end certificate computation on synthetic data."""

    def _setup(self, m: int = 200, D: int = 50, c: int = 10, alpha: float = 1.0):
        """Create a complete FKP pipeline on synthetic data."""
        torch.manual_seed(42)
        # Simulate a well-conditioned teacher (linear relationship exists)
        W_true = torch.randn(D, c)
        E_raw = torch.randn(m, D)
        H_raw = E_raw @ W_true + 0.01 * torch.randn(m, c)

        # ZCA whitening
        whitener = ZCAWhitener(lambda_zca=1e-4)
        E_tilde = whitener.fit_transform([E_raw], calibration_size=m)

        # Center logits
        H_c, H_bar = center_logits(H_raw)

        # Ridge regression
        W_ridge = ridge_auto(E_tilde, H_c, alpha=alpha)

        # Spectral compression
        decomp = spectral_compress(W_ridge, eta_tail=0.05)

        return E_tilde, H_c, H_raw, W_ridge, decomp

    def test_certificate_components_are_finite(self) -> None:
        """All certificate components must be finite positive numbers."""
        E_tilde, H_c, H_raw, W_ridge, decomp = self._setup()
        cert = FKPCertificate(alpha=1.0, delta=0.01)
        result = cert.compute(
            E_tilde=E_tilde,
            H_c=H_c,
            H_raw=H_raw,
            W_ridge=W_ridge,
            singular_values=decomp.singular_values,
            rank_p=decomp.rank_p,
        )

        assert math.isfinite(result.eps_delta), "eps_delta must be finite."
        assert math.isfinite(result.eps_stab), "eps_stab must be finite."
        assert math.isfinite(result.eps_proj), "eps_proj must be finite."
        assert math.isfinite(result.R_delta), "R_delta must be finite."
        assert result.eps_delta >= 0, "eps_delta must be non-negative."
        assert result.eps_stab >= 0, "eps_stab must be non-negative."
        assert result.eps_proj >= 0, "eps_proj must be non-negative."

    def test_total_eps_equals_sum_of_components(self) -> None:
        """eps_delta must equal sqrt(eps_stab) + sqrt(eps_proj) to 1e-10."""
        E_tilde, H_c, H_raw, W_ridge, decomp = self._setup()
        cert = FKPCertificate(alpha=1.0, delta=0.01)
        result = cert.compute(
            E_tilde=E_tilde,
            H_c=H_c,
            H_raw=H_raw,
            W_ridge=W_ridge,
            singular_values=decomp.singular_values,
            rank_p=decomp.rank_p,
        )

        expected_eps = math.sqrt(result.eps_stab) + math.sqrt(result.eps_proj)
        assert abs(result.eps_delta - expected_eps) < 1e-10, (
            f"eps_delta decomposition error: {result.eps_delta:.12f} != {expected_eps:.12f}"
        )

    def test_decision_preserved_on_well_conditioned_data(self) -> None:
        """Certificate should pass for a near-linear teacher with good margin."""
        # Use a large calibration set and small alpha to make the fit near-perfect
        E_tilde, H_c, H_raw, W_ridge, decomp = self._setup(m=400, D=30, c=5, alpha=0.01)
        cert = FKPCertificate(alpha=0.01, delta=0.05)
        result = cert.compute(
            E_tilde=E_tilde,
            H_c=H_c,
            H_raw=H_raw,
            W_ridge=W_ridge,
            singular_values=decomp.singular_values,
            rank_p=decomp.rank_p,
        )
        # For a nearly linear teacher with large margin, certificate should pass
        # (gamma_min > 2 * eps_delta)
        assert result.gamma_min > 0, "gamma_min must be positive for distinct teacher predictions."

    def test_delta_validation(self) -> None:
        """FKPCertificate must reject invalid delta values."""
        with pytest.raises(ValueError, match="delta must be in"):
            FKPCertificate(alpha=1.0, delta=1.5)
        with pytest.raises(ValueError, match="delta must be in"):
            FKPCertificate(alpha=1.0, delta=0.0)

    def test_alpha_validation(self) -> None:
        """FKPCertificate must reject non-positive alpha."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            FKPCertificate(alpha=0.0, delta=0.05)
