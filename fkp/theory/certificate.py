"""
FKP High-Probability Decision Certificate.

Combines the stability bound, feature norm concentration, and projection
loss bound (all from bounds.py) into the unified certificate from
Theorem 1 and Corollary 1 of the FKP paper:

    eps_delta = sqrt(eps_stab) + sqrt(eps_proj)

Decision is guaranteed to be preserved if:
    eps_delta < gamma_min / 2

where gamma_min is the teacher's minimum logit margin on the calibration set.

Reference:
    §3.5 "Certificate" and §4.2 "Finite-Sample Certificate" in FKP.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from fkp.theory.bounds import (
    stability_bound,
    feature_norm_bound,
    projection_loss_bound,
)


@dataclass
class CertificateResult:
    """Full output of the FKP certificate computation.

    Attributes
    ----------
    eps_delta : float
        Total logit approximation error bound:
        eps_delta = sqrt(eps_stab) + sqrt(eps_proj)
    eps_stab : float
        Stability component of the bound.
    eps_proj : float
        Projection loss component of the bound.
    R_delta : float
        Feature norm concentration bound.
    gamma_min : float
        Teacher's minimum logit margin over the calibration set.
    decision_preserved : bool
        True if eps_delta < gamma_min / 2 (Corollary 1).
    delta : float
        Confidence level; guarantee holds with probability >= 1 - delta.
    empirical_error : float or None
        Empirical ||M_theta(x) - f_FKP(x)||_2 averaged over calibration set,
        if computable.  None if not available.
    bound_ratio : float or None
        eps_delta / empirical_error.  Measures certificate conservativeness.
    """
    eps_delta: float
    eps_stab: float
    eps_proj: float
    R_delta: float
    gamma_min: float
    decision_preserved: bool
    delta: float
    empirical_error: float | None = None
    bound_ratio: float | None = None

    def __str__(self) -> str:
        badge = "CERTIFICATE PASSED ✓" if self.decision_preserved else "CERTIFICATE FAILED ✗"
        lines = [
            "=" * 60,
            f"  FKP Certificate  (delta={self.delta:.4f})",
            "=" * 60,
            f"  eps_stab          = {self.eps_stab:.6f}",
            f"  eps_proj          = {self.eps_proj:.6f}",
            f"  R_delta           = {self.R_delta:.6f}",
            f"  eps_delta (total) = {self.eps_delta:.6f}",
            f"  gamma_min / 2     = {self.gamma_min / 2:.6f}",
            f"  Condition met     : {self.eps_delta:.6f} < {self.gamma_min / 2:.6f}",
            f"  {badge}",
        ]
        if self.empirical_error is not None:
            lines.append(f"  Empirical error   = {self.empirical_error:.6f}")
        if self.bound_ratio is not None:
            lines.append(f"  Bound ratio       = {self.bound_ratio:.2f}x")
        lines.append("=" * 60)
        return "\n".join(lines)


class FKPCertificate:
    """Computes and verifies the FKP high-probability decision certificate.

    Usage
    -----
    cert = FKPCertificate(alpha=1.0, delta=0.01)
    result = cert.compute(
        E_tilde=E_tilde,
        H_c=H_c,
        H_raw=H_raw,
        W_ridge=W_ridge,
        singular_values=singular_values,
        rank_p=p,
    )
    print(result)

    Parameters
    ----------
    alpha : float
        Ridge regularization penalty used during training.
    delta : float
        Failure probability.  Certificate holds with probability >= 1 - delta.
    sigma_E : float or None
        Sub-Gaussian parameter for feature norm bound.  If None, estimated
        as sqrt(D).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        delta: float = 0.01,
        sigma_E: float | None = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not (0 < delta < 1):
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        self.alpha = alpha
        self.delta = delta
        self.sigma_E = sigma_E

    def compute(
        self,
        E_tilde: torch.Tensor,
        H_c: torch.Tensor,
        H_raw: torch.Tensor,
        W_ridge: torch.Tensor,
        singular_values: torch.Tensor,
        rank_p: int,
        logits_edge: torch.Tensor | None = None,
    ) -> CertificateResult:
        """Compute the complete certificate.

        Parameters
        ----------
        E_tilde : torch.Tensor
            Conditioned feature design matrix of shape (m, D).
        H_c : torch.Tensor
            Centered teacher logit matrix of shape (m, c).
        H_raw : torch.Tensor
            Raw (un-centered) teacher logit matrix of shape (m, c).
            Used to compute gamma_min.
        W_ridge : torch.Tensor
            Full ridge weight matrix of shape (D, c).
        singular_values : torch.Tensor
            All singular values of W_ridge in descending order, shape (r,).
        rank_p : int
            Selected projection rank p.
        logits_edge : torch.Tensor or None
            Edge model predictions on calibration set, shape (m, c).
            If provided, empirical error and bound ratio are computed.

        Returns
        -------
        CertificateResult
            Complete certificate with all components.
        """
        # --- Stability bound ---
        stab = stability_bound(E_tilde, H_c, W_ridge, self.alpha, self.delta)

        # --- Feature norm bound ---
        R_delta = feature_norm_bound(E_tilde, self.delta, self.sigma_E)

        # --- Projection loss bound ---
        eps_proj = projection_loss_bound(singular_values, rank_p, R_delta)

        # --- Total certificate (Theorem 1) ---
        eps_delta = math.sqrt(stab.eps_stab) + math.sqrt(eps_proj)

        # --- Minimum logit margin (Corollary 1) ---
        gamma_min = self._compute_gamma_min(H_raw)

        # --- Decision preservation check ---
        decision_preserved = eps_delta < gamma_min / 2.0

        # --- Empirical error (if edge logits available) ---
        empirical_error = None
        bound_ratio = None
        if logits_edge is not None:
            H_bar = H_raw.mean(dim=0, keepdim=True)
            # Full ridge prediction for comparison
            H_hat_full = E_tilde.to(W_ridge.dtype) @ W_ridge + H_bar.to(W_ridge.dtype)
            emp_err = (
                (H_raw.to(torch.float32) - logits_edge.to(torch.float32))
                .norm(dim=1)
                .mean()
                .item()
            )
            empirical_error = emp_err
            if emp_err > 1e-12:
                bound_ratio = eps_delta / emp_err

        return CertificateResult(
            eps_delta=eps_delta,
            eps_stab=stab.eps_stab,
            eps_proj=eps_proj,
            R_delta=R_delta,
            gamma_min=gamma_min,
            decision_preserved=decision_preserved,
            delta=self.delta,
            empirical_error=empirical_error,
            bound_ratio=bound_ratio,
        )

    def _compute_gamma_min(self, H_raw: torch.Tensor) -> float:
        """Compute the minimum teacher logit margin over the calibration set.

        gamma_min = min_i [ max_class logit_i - second_max logit_i ]

        This corresponds to the teacher's confidence margin — the gap between
        the top logit and the runner-up for each calibration sample.

        Parameters
        ----------
        H_raw : torch.Tensor
            Teacher logit matrix of shape (m, c).

        Returns
        -------
        gamma_min : float
            Minimum margin.  If all samples have the same top-2 logits,
            returns 0.0 (worst case — certificate always passes vacuously).
        """
        H = H_raw.to(torch.float64)
        # Top-2 logits per sample
        top2, _ = H.topk(2, dim=1)    # (m, 2)
        margins = top2[:, 0] - top2[:, 1]  # (m,)
        return margins.min().item()
