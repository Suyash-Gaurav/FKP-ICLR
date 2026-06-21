"""
Sub-Gaussian Norm Bounds and Uniform Stability Bounds for FKP.

This module implements the three theoretical bounds that compose the
high-probability certificate (Theorem 1 in the FKP paper):

    1. Stability Bound (eps_stab):
       Uniform stability of multi-output ridge regression bounds the logit
       approximation error of the full-rank ridge predictor.

    2. Feature Norm Bound (R_delta):
       Sub-Gaussian concentration bounds the L2 norm of the conditioned
       feature vector at inference time.

    3. Projection Loss Bound (eps_proj):
       Spectral perturbation theory bounds the additional error introduced
       by truncating to rank p.

Reference:
    §3.5 "Certificate: High-Probability Error Guarantee" and
    §4.2 "Finite-Sample Certificate" in the FKP paper.
    [Bousquet & Elisseeff, 2002] for uniform stability.
    [Wainwright, 2019] for sub-Gaussian concentration.
    [Stewart & Sun, 1990] for spectral perturbation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class StabilityBoundResult:
    """Components of the stability bound epsilon_stab.

    Attributes
    ----------
    eps_stab : float
        Total stability bound value.  sqrt(eps_stab) enters the certificate.
    E_reg : float
        Empirical ridge regression loss: (1/m) ||H_c - E W||_F^2.
    gamma : float
        Regularization penalty term: 2*alpha*||W||_F^2 / m.
    c_delta : float
        Confidence coefficient: sqrt(2*log(4/delta) / m).
    """
    eps_stab: float
    E_reg: float
    gamma: float
    c_delta: float


def stability_bound(
    E_tilde: torch.Tensor,
    H_c: torch.Tensor,
    W_ridge: torch.Tensor,
    alpha: float,
    delta: float,
) -> StabilityBoundResult:
    """Compute the uniform stability bound for multi-output ridge regression.

    Multi-output ridge regression is uniformly stable with parameter
    beta = 2 / (m * alpha) [Bousquet & Elisseeff, 2002].  Applying the
    McDiarmid / stability-based generalization bound gives:

        P[ ||hat_y_full(x) - M_theta(x)||_2^2 <= eps_stab ] >= 1 - delta/2

    where:
        eps_stab  = E_reg + gamma + c_{delta/2} * (E_reg + gamma)
        E_reg     = (1/m) ||H_c - E W_ridge||_F^2
        gamma     = 2 * alpha * ||W_ridge||_F^2 / m
        c_{delta/2} = sqrt(2 * log(4/delta) / m)

    Parameters
    ----------
    E_tilde : torch.Tensor
        Conditioned feature design matrix of shape (m, D).
    H_c : torch.Tensor
        Centered teacher logit matrix of shape (m, c).
    W_ridge : torch.Tensor
        Ridge solution weight matrix of shape (D, c).
    alpha : float
        Ridge penalty used in training.
    delta : float
        Failure probability.  Bound holds with probability >= 1 - delta/2.

    Returns
    -------
    StabilityBoundResult
        All components of eps_stab.
    """
    _validate_delta(delta)
    m = E_tilde.shape[0]
    E = E_tilde.to(torch.float64)
    H = H_c.to(torch.float64)
    W = W_ridge.to(torch.float64)

    # Empirical ridge loss
    residual = H - E @ W                          # (m, c)
    E_reg = (residual.pow(2).sum() / m).item()

    # Regularization penalty
    gamma = (2.0 * alpha * W.pow(2).sum() / m).item()

    # Confidence coefficient (at delta/2 for union bound)
    c_delta = math.sqrt(2.0 * math.log(4.0 / delta) / m)

    eps_stab = E_reg + gamma + c_delta * (E_reg + gamma)

    return StabilityBoundResult(
        eps_stab=eps_stab,
        E_reg=E_reg,
        gamma=gamma,
        c_delta=c_delta,
    )


def feature_norm_bound(
    E_tilde: torch.Tensor,
    delta: float,
    sigma_E: float | None = None,
) -> float:
    """Sub-Gaussian feature norm concentration bound R_delta.

    Under the sub-Gaussian assumption (Assumption 1 in the paper),
    with probability >= 1 - delta/2:

        ||tilde_E(x)||_2 <= sigma_E * (1 + sqrt(2*log(4/delta)/m)
                                         + 2*log(4/delta)/m )
                         := R_delta

    When sigma_E is not provided, it is estimated from the calibration
    set as the empirical sub-Gaussian proxy:
        sigma_E_hat = max_k sqrt(mean(||tilde_phi_k(x)||_2^2 / d_k))

    For ZCA-whitened features with identity covariance, the standard
    deviation per coordinate is 1, so sigma_E is approximated by
    sqrt(D) where D is the total dimension.

    Parameters
    ----------
    E_tilde : torch.Tensor
        Conditioned feature design matrix of shape (m, D).
    delta : float
        Failure probability.  Bound holds with probability >= 1 - delta/2.
    sigma_E : float or None
        Sub-Gaussian parameter.  If None, estimated as sqrt(D) from the
        identity-covariance property of ZCA-whitened features.

    Returns
    -------
    R_delta : float
        Feature norm concentration bound.
    """
    _validate_delta(delta)
    m, D = E_tilde.shape

    if sigma_E is None:
        # For ZCA-whitened features: E[||tilde_E||_2^2] = D (trace of identity)
        # => sigma_E ~ sqrt(D) is the natural scale
        sigma_E = math.sqrt(float(D))

    log_term = math.log(4.0 / delta)
    R_delta = sigma_E * (
        1.0 + math.sqrt(2.0 * log_term / m) + 2.0 * log_term / m
    )
    return R_delta


def projection_loss_bound(
    singular_values: torch.Tensor,
    rank_p: int,
    R_delta: float,
) -> float:
    """Spectral perturbation bound on the projection loss (eps_proj).

    Using the Eckart-Young-Mirsky theorem and spectral norm perturbation:

        ||hat_y_edge(x) - hat_y_full(x)||_2
            <= ||W_ridge - W_FKP||_2 * ||tilde_E(x)||_2
            = sigma_{p+1} * ||tilde_E(x)||_2
            <= sigma_{p+1} * R_delta

    Hence:
        eps_proj = sigma_{p+1}^2 * R_delta^2

    Parameters
    ----------
    singular_values : torch.Tensor
        All singular values of W_ridge in descending order, shape (r,).
    rank_p : int
        Selected projection rank p.  The (p+1)-th singular value
        sigma_{p+1} is the spectral norm of (W_ridge - W_FKP).
    R_delta : float
        Feature norm concentration bound from feature_norm_bound().

    Returns
    -------
    eps_proj : float
        Projection loss bound value.
    """
    r = len(singular_values)
    if rank_p >= r:
        # Exact representation — zero projection error
        return 0.0

    sigma_next = singular_values[rank_p].item()   # sigma_{p+1} (0-indexed)
    eps_proj = (sigma_next ** 2) * (R_delta ** 2)
    return eps_proj


def _validate_delta(delta: float) -> None:
    if not (0 < delta < 1):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
