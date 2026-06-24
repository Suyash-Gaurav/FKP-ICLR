"""
Closed-form multi-output Ridge Regression — Primal and Dual forms.

IMPORTANT: This module deliberately does NOT use sklearn or torch.nn.Linear.
All computations use torch.linalg directly so that the exact dual-form
inversion is transparent and mathematically verifiable, as required by the
paper's methodology section.

Equations implemented:
    Primal (Eq. 5):  W = (E^T E + m*alpha*I_D)^{-1} E^T H_c     [O(D^3)]
    Dual   (Eq. 6):  W = E^T (E E^T + m*alpha*I_m)^{-1} H_c     [O(m^3)]

The dual form is used when D > m to reduce inversion cost from O(D^3) to
O(m^3).  ridge_auto() selects the appropriate form automatically.

Reference:
    §3.1 "Unified Framework" and §3.4 "Ridge Regression Projection" in FKP.
"""

from __future__ import annotations

import torch
import torch.linalg as LA

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ridge_primal(
    E: torch.Tensor,
    H: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Primal closed-form ridge regression.

    Solves:  min_W  (1/2m)||H_c - E W||_F^2 + (alpha/2)||W||_F^2

    Closed-form solution (Eq. 5):
        W = (E^T E + m*alpha*I_D)^{-1} E^T H_c

    Note: H is assumed already centered (H_c = H - 1*H_bar^T).

    Parameters
    ----------
    E : torch.Tensor
        Conditioned feature design matrix of shape (m, D).
        Should be ZCA-whitened (zero mean, ~identity covariance).
    H : torch.Tensor
        Centered teacher logit matrix of shape (m, c).
    alpha : float
        Ridge regularization penalty (> 0).

    Returns
    -------
    W : torch.Tensor
        Weight matrix of shape (D, c).

    Raises
    ------
    ValueError
        If shapes are inconsistent or alpha is non-positive.
    """
    _validate_inputs(E, H, alpha, form="primal")
    m, D = E.shape
    dtype = torch.float64

    E = E.to(dtype=dtype)
    H = H.to(dtype=dtype)

    # Gram matrix: E^T E  (D, D)
    EtE = E.T @ E                                               # (D, D)
    reg = (m * alpha) * torch.eye(D, dtype=dtype, device=E.device)
    A = EtE + reg                                               # (D, D)

    # Solve the system (D, D) @ W = E^T H  =>  W = A^{-1} (E^T H)
    EtH = E.T @ H                                               # (D, c)
    W = LA.solve(A, EtH)                                        # (D, c)
    return W.to(torch.float32)


def ridge_dual(
    E: torch.Tensor,
    H: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Dual closed-form ridge regression (efficient when D > m).

    Uses the kernel-side inversion (Eq. 6):
        W = E^T (E E^T + m*alpha*I_m)^{-1} H_c

    This reduces the matrix inversion from O(D^3) to O(m^3), which is
    critical when multi-layer ViT features have D >> m.

    Parameters
    ----------
    E : torch.Tensor
        Conditioned feature design matrix of shape (m, D).
    H : torch.Tensor
        Centered teacher logit matrix of shape (m, c).
    alpha : float
        Ridge regularization penalty (> 0).

    Returns
    -------
    W : torch.Tensor
        Weight matrix of shape (D, c).  Mathematically identical to the
        primal solution; see test_ridge_dual.py for numerical verification.
    """
    _validate_inputs(E, H, alpha, form="dual")
    m, D = E.shape
    dtype = torch.float64

    E = E.to(dtype=dtype)
    H = H.to(dtype=dtype)

    # Kernel (Gram) matrix: E E^T  (m, m)
    EEt = E @ E.T                                                # (m, m)
    reg = (m * alpha) * torch.eye(m, dtype=dtype, device=E.device)
    K = EEt + reg                                                # (m, m)

    # Dual coefficients: alpha_mat = K^{-1} H_c   (m, c)
    alpha_mat = LA.solve(K, H)                                   # (m, c)

    # Primal weights via dual:  W = E^T alpha_mat  (D, c)
    W = E.T @ alpha_mat                                          # (D, c)
    return W.to(torch.float32)


def ridge_auto(
    E: torch.Tensor,
    H: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Automatically select primal or dual ridge regression based on D vs m.

    Uses the dual form when D > m (cheaper inversion) and the primal form
    otherwise.  Both forms produce mathematically identical results.

    Parameters
    ----------
    E : torch.Tensor
        Conditioned feature design matrix of shape (m, D).
    H : torch.Tensor
        Centered teacher logit matrix of shape (m, c).
    alpha : float
        Ridge regularization penalty (> 0).

    Returns
    -------
    W : torch.Tensor
        Weight matrix of shape (D, c).
    """
    m, D = E.shape
    if D > m:
        return ridge_dual(E, H, alpha)
    return ridge_primal(E, H, alpha)


def _validate_inputs(E: torch.Tensor, H: torch.Tensor, alpha: float, form: str) -> None:
    """Shared input validation for ridge functions."""
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D (m, D), got shape {tuple(E.shape)}")
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D (m, c), got shape {tuple(H.shape)}")
    if E.shape[0] != H.shape[0]:
        raise ValueError(
            f"E and H must have the same number of rows (calibration samples). "
            f"Got E: {E.shape[0]}, H: {H.shape[0]}"
        )
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    if form == "primal" and E.shape[1] < E.shape[0]:
        pass  # valid, no warning needed
