"""
Unit test: ZCA whitening produces identity covariance.

This is the critical math test described in Phase 2 of the intern guide:
    "A unit test for FKP isn't 'does the code run without crashing.'
     A unit test is 'does the empirical covariance of the ZCA-whitened
     features equal the Identity matrix within 1e-4 tolerance?'"

Tests cover:
    1. Standard case (D < m): empirical covariance ≈ I
    2. PCA-ZCA hybrid case (D > m): empirical covariance ≈ I in reduced space
    3. Zero-mean property after whitening
    4. transform() consistency with fit_transform()
"""

import pytest
import torch

from fkp.conditioning.zca import ZCAWhitener


def _make_correlated_features(m: int, d: int, seed: int = 0) -> torch.Tensor:
    """Generate correlated (non-whitened) Gaussian features."""
    torch.manual_seed(seed)
    # Random covariance: A^T A
    A = torch.randn(d, d)
    cov = A.T @ A / d + 0.1 * torch.eye(d)
    L = torch.linalg.cholesky(cov)
    raw = torch.randn(m, d)
    return raw @ L.T


class TestZCAStandardCase:
    """Tests for D < m (no PCA-ZCA hybrid needed)."""

    def test_empirical_covariance_is_identity(self) -> None:
        """After ZCA whitening, empirical covariance must equal I within 1e-4.

        This is the central mathematical guarantee of ZCA whitening.
        By construction:  (1/m) * E_tilde^T * E_tilde ≈ I_D
        """
        m, d = 500, 50
        phi = _make_correlated_features(m, d, seed=0)
        whitener = ZCAWhitener(lambda_zca=1e-4)
        E_tilde = whitener.fit_transform([phi], calibration_size=m)  # (m, d)

        # Empirical covariance of whitened features
        E_c = E_tilde - E_tilde.mean(dim=0, keepdim=True)
        C_emp = (E_c.T @ E_c) / m

        identity = torch.eye(d)
        max_off_diag = (C_emp - identity).abs().max().item()

        assert max_off_diag < 1e-4, (
            f"ZCA whitening failed: max |C_emp - I| = {max_off_diag:.2e} "
            f"(expected < 1e-4).  Check ZCA computation."
        )

    def test_zero_mean_after_whitening(self) -> None:
        """Whitened features must have approximately zero mean."""
        m, d = 300, 30
        phi = _make_correlated_features(m, d, seed=1)
        whitener = ZCAWhitener(lambda_zca=1e-4)
        E_tilde = whitener.fit_transform([phi], calibration_size=m)

        mean_norm = E_tilde.mean(dim=0).abs().max().item()
        assert mean_norm < 1e-6, (
            f"ZCA-whitened features have non-zero mean: max |mean| = {mean_norm:.2e}"
        )

    def test_multilayer_concatenation_shape(self) -> None:
        """Output shape must be (m, sum_k d_k)."""
        m = 200
        dims = [32, 64, 128]
        layers = [_make_correlated_features(m, d, seed=i) for i, d in enumerate(dims)]

        whitener = ZCAWhitener(lambda_zca=1e-4)
        E_tilde = whitener.fit_transform(layers, calibration_size=m)

        expected_D = sum(dims)
        assert E_tilde.shape == (m, expected_D), (
            f"Expected shape ({m}, {expected_D}), got {tuple(E_tilde.shape)}"
        )

    def test_transform_matches_fit_transform(self) -> None:
        """transform() after fit() must equal fit_transform()."""
        m, d = 200, 40
        phi = _make_correlated_features(m, d, seed=2)
        layers = [phi]

        whitener_a = ZCAWhitener(lambda_zca=1e-4)
        E_ft = whitener_a.fit_transform(layers, calibration_size=m)

        whitener_b = ZCAWhitener(lambda_zca=1e-4)
        whitener_b.fit(layers, calibration_size=m)
        E_t = whitener_b.transform(layers)

        max_diff = (E_ft - E_t).abs().max().item()
        assert max_diff < 1e-7, (
            f"fit_transform() and fit() + transform() disagree: {max_diff:.2e}"
        )


class TestZCAPCAHybrid:
    """Tests for D > m (PCA-ZCA hybrid must be triggered)."""

    def test_pca_zca_triggered_when_d_greater_m(self) -> None:
        """When d > m for a layer, used_pca must be True."""
        m, d = 100, 500   # D > m
        torch.manual_seed(3)
        phi = torch.randn(m, d)  # rank-deficient covariance
        whitener = ZCAWhitener(lambda_zca=1e-4, pca_variance_threshold=0.99)
        whitener.fit([phi], calibration_size=m)
        assert whitener.layer_states[0].used_pca, (
            "PCA-ZCA hybrid was not triggered despite d > m."
        )

    def test_pca_zca_output_covariance_is_identity(self) -> None:
        """PCA-ZCA hybrid must also produce near-identity empirical covariance."""
        m, d = 100, 500
        torch.manual_seed(4)
        phi = torch.randn(m, d)
        whitener = ZCAWhitener(lambda_zca=1e-4, pca_variance_threshold=0.99)
        E_tilde = whitener.fit_transform([phi], calibration_size=m)

        D_red = E_tilde.shape[1]
        E_c = E_tilde - E_tilde.mean(dim=0, keepdim=True)
        C_emp = (E_c.T @ E_c) / m
        identity = torch.eye(D_red)
        max_off_diag = (C_emp - identity).abs().max().item()

        assert max_off_diag < 1e-3, (
            f"PCA-ZCA whitening failed: max |C_emp - I| = {max_off_diag:.2e} "
            f"(PCA-reduced dim = {D_red})"
        )

    def test_pca_reduces_dimension(self) -> None:
        """PCA-ZCA should reduce d to something strictly less than d."""
        m, d = 80, 500
        torch.manual_seed(5)
        phi = torch.randn(m, d)
        whitener = ZCAWhitener(lambda_zca=1e-4, pca_variance_threshold=0.99)
        whitener.fit([phi], calibration_size=m)
        state = whitener.layer_states[0]
        assert state.d_reduced < d, (
            f"PCA-ZCA did not reduce dimension: d_original={d}, d_reduced={state.d_reduced}"
        )
        # Reduced dim must be <= m (since rank(phi) <= min(m, d) = m)
        assert state.d_reduced <= m, (
            f"d_reduced={state.d_reduced} exceeds rank upper bound m={m}"
        )


class TestZCAValidation:
    """Error handling tests."""

    def test_unfitted_raises(self) -> None:
        whitener = ZCAWhitener()
        with pytest.raises(RuntimeError, match="must be fitted"):
            whitener.transform([torch.randn(10, 5)])

    def test_layer_count_mismatch_raises(self) -> None:
        m, d = 50, 10
        phi = torch.randn(m, d)
        whitener = ZCAWhitener()
        whitener.fit([phi], calibration_size=m)
        with pytest.raises(ValueError, match="Expected 1 layers, got 2"):
            whitener.transform([phi, phi])

    def test_negative_lambda_raises(self) -> None:
        with pytest.raises(ValueError, match="lambda_zca must be positive"):
            ZCAWhitener(lambda_zca=-1e-4)
