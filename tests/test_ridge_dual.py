"""
Unit test: Primal and Dual ridge regression produce identical weight matrices.

This is the critical math test described in Phase 2 of the intern guide:
    "Generate random Gaussian data where D=2000 and m=500.  Prove mathematically
     in the test that the Primal and Dual weight matrices are identical within
     a tolerance of 1e-5."

A unit test for FKP isn't 'does the code run without crashing.'
A unit test is 'do the two closed-form solutions produce the same result?'
"""

import pytest
import torch

from fkp.compression.ridge import ridge_primal, ridge_dual, ridge_auto


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_primal_dual_equivalence_d_greater_m(alpha: float) -> None:
    """Primal and dual forms must agree when D > m (1e-5 tolerance).

    Setup: D=2000, m=500, c=10.
    This is the canonical case where the dual form is computationally
    preferred.  Mathematical identity (the matrix inversion lemma) guarantees
    the two forms produce the same weights up to floating-point precision.
    """
    torch.manual_seed(0)
    m, D, c = 500, 2000, 10

    E = torch.randn(m, D)
    H = torch.randn(m, c)

    W_primal = ridge_primal(E, H, alpha=alpha)
    W_dual = ridge_dual(E, H, alpha=alpha)

    assert W_primal.shape == (D, c), f"Expected (D,c)=({D},{c}), got {W_primal.shape}"
    assert W_dual.shape == (D, c), f"Expected (D,c)=({D},{c}), got {W_dual.shape}"

    max_diff = (W_primal - W_dual).abs().max().item()
    assert max_diff < 1e-5, (
        f"Primal and dual solutions disagree: max |W_primal - W_dual| = {max_diff:.2e} "
        f"(alpha={alpha}, D={D}, m={m}).  This indicates a numerical or algorithmic bug."
    )


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0])
def test_primal_dual_equivalence_d_less_m(alpha: float) -> None:
    """Primal and dual forms must also agree when D < m."""
    torch.manual_seed(1)
    m, D, c = 500, 100, 5

    E = torch.randn(m, D)
    H = torch.randn(m, c)

    W_primal = ridge_primal(E, H, alpha=alpha)
    W_dual = ridge_dual(E, H, alpha=alpha)

    max_diff = (W_primal - W_dual).abs().max().item()
    assert max_diff < 1e-5, (
        f"Primal and dual disagree: max |W_primal - W_dual| = {max_diff:.2e} "
        f"(alpha={alpha})"
    )


def test_ridge_auto_selects_dual_when_d_greater_m() -> None:
    """ridge_auto must select dual form when D > m and produce the correct result."""
    torch.manual_seed(2)
    m, D, c = 200, 800, 10

    E = torch.randn(m, D)
    H = torch.randn(m, c)

    W_auto = ridge_auto(E, H, alpha=1.0)
    W_dual = ridge_dual(E, H, alpha=1.0)

    max_diff = (W_auto - W_dual).abs().max().item()
    assert max_diff < 1e-7, (
        f"ridge_auto should match ridge_dual when D > m.  Diff: {max_diff:.2e}"
    )


def test_ridge_auto_selects_primal_when_d_le_m() -> None:
    """ridge_auto must select primal form when D <= m."""
    torch.manual_seed(3)
    m, D, c = 500, 200, 10

    E = torch.randn(m, D)
    H = torch.randn(m, c)

    W_auto = ridge_auto(E, H, alpha=1.0)
    W_primal = ridge_primal(E, H, alpha=1.0)

    max_diff = (W_auto - W_primal).abs().max().item()
    assert max_diff < 1e-7, (
        f"ridge_auto should match ridge_primal when D <= m.  Diff: {max_diff:.2e}"
    )


def test_ridge_recovers_true_weights_noiseless() -> None:
    """Ridge regression should recover true weights when noise is zero and alpha -> 0."""
    torch.manual_seed(4)
    m, D, c = 300, 50, 5

    W_true = torch.randn(D, c)
    E = torch.randn(m, D)
    H = E @ W_true   # noiseless observations

    # With very small alpha, should approximate true weights
    W_est = ridge_primal(E, H, alpha=1e-8)
    max_diff = (W_est - W_true).abs().max().item()
    assert max_diff < 1e-3, (
        f"Ridge failed to recover true weights in noiseless case.  Max diff: {max_diff:.2e}"
    )


def test_ridge_input_validation() -> None:
    """ridge functions must raise ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="alpha must be positive"):
        ridge_primal(torch.randn(10, 5), torch.randn(10, 3), alpha=-1.0)

    with pytest.raises(ValueError):
        ridge_dual(torch.randn(10, 5), torch.randn(12, 3), alpha=1.0)  # row mismatch

    with pytest.raises(ValueError):
        ridge_primal(torch.randn(3, 5, 2), torch.randn(3, 2), alpha=1.0)  # 3-D E
