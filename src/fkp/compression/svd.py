"""
Truncated SVD and Spectral Projection for FKP.

Implements the spectral compression stage that converts the full ridge
weight matrix W_ridge into the ultra-compact edge representation:

    W_ridge = U Sigma V^T   (compact SVD)
    W_FKP   = U_p Sigma_p V_p^T   (rank-p truncation)

The edge device stores only:
    W_edge = Sigma_p V_p^T  in R^{p x c}
    b      = H_bar          in R^c
    U_p    = projection basis in R^{D x p}  (stays on gateway)

Reference:
    §3.3 "Spectral Projection (FKP)" and §3.4 "Spectral Compression" in FKP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.linalg as LA


@dataclass
class SpectralDecomposition:
    """Result of the spectral compression stage.

    Attributes
    ----------
    U_p : torch.Tensor
        Left singular vectors of shape (D, p).  Stored on the gateway.
        Used to project the conditioned feature z = U_p^T @ tilde_E.
    W_edge : torch.Tensor
        Edge weight matrix Sigma_p V_p^T of shape (p, c).  Stored on MCU.
    singular_values : torch.Tensor
        All singular values of W_ridge in descending order, shape (r,).
    rank_p : int
        Selected rank p.
    tail_energy_ratio : float
        Fraction of spectral energy in the discarded tail:
        sum_{i>p} sigma_i^2 / ||Sigma||_F^2.
    W_ridge : torch.Tensor
        The original full-rank ridge weight matrix (D, c), retained for
        certificate computation.
    """
    U_p: torch.Tensor
    W_edge: torch.Tensor
    singular_values: torch.Tensor
    rank_p: int
    tail_energy_ratio: float
    W_ridge: torch.Tensor


def select_rank(
    singular_values: torch.Tensor,
    eta_tail: float = 0.05,
) -> int:
    """Select the minimum rank p such that the spectral tail is below eta_tail.

    Formally: find smallest p such that
        sum_{i=p+1}^r sigma_i^2 / ||Sigma||_F^2  <  eta_tail

    Parameters
    ----------
    singular_values : torch.Tensor
        Singular values in descending order, shape (r,).
    eta_tail : float
        Tail energy threshold.  Default: 0.05 (retains >=95% spectral energy).

    Returns
    -------
    p : int
        Optimal projection rank.  Always at least 1.
    """
    if eta_tail <= 0 or eta_tail >= 1:
        raise ValueError(f"eta_tail must be in (0, 1), got {eta_tail}")

    energy = singular_values.pow(2)
    total_energy = energy.sum()
    if total_energy == 0:
        return 1

    # Cumulative energy from left (kept) side
    cumulative_kept = energy.cumsum(dim=0)
    # For rank p: tail_ratio = 1 - cumulative_kept[p-1] / total
    for p in range(1, len(singular_values) + 1):
        tail_ratio = 1.0 - (cumulative_kept[p - 1] / total_energy).item()
        if tail_ratio < eta_tail:
            return p

    # All ranks needed (tail never drops below threshold)
    return len(singular_values)


def spectral_compress(
    W_ridge: torch.Tensor,
    eta_tail: float = 0.05,
    rank_p: int | None = None,
) -> SpectralDecomposition:
    """Compress the ridge weight matrix via truncated SVD.

    Steps:
      1. Compute the compact SVD: W_ridge = U Sigma V^T
      2. Select rank p (auto or user-specified)
      3. Truncate to top-p components:
           U_p = U[:, :p]          (D, p)
           Sigma_p = Sigma[:p]
           V_p = V[:, :p]          (c, p)
      4. Form edge weight: W_edge = diag(Sigma_p) @ V_p^T  (p, c)

    Parameters
    ----------
    W_ridge : torch.Tensor
        Full ridge weight matrix of shape (D, c).
    eta_tail : float
        Tail energy threshold for automatic rank selection.  Ignored if
        rank_p is provided explicitly.
    rank_p : int or None
        If given, forces the projection rank to exactly this value.
        Must be in [1, min(D, c)].  If None, rank is selected automatically.

    Returns
    -------
    SpectralDecomposition
        Compact SVD result with all quantities needed for edge deployment
        and certificate computation.
    """
    if W_ridge.ndim != 2:
        raise ValueError(f"W_ridge must be 2-D (D, c), got shape {tuple(W_ridge.shape)}")

    D, c = W_ridge.shape
    r = min(D, c)

    W = W_ridge.to(dtype=torch.float64)

    # Compact SVD: U (D, r), S (r,), Vh (r, c)
    U, S, Vh = LA.svd(W, full_matrices=False)
    # S is already in descending order (guaranteed by torch.linalg.svd)

    # --- Rank selection ---
    if rank_p is None:
        p = select_rank(S, eta_tail=eta_tail)
    else:
        if not (1 <= rank_p <= r):
            raise ValueError(
                f"rank_p must be in [1, min(D,c)]=[1,{r}], got {rank_p}"
            )
        p = rank_p

    # Truncate
    U_p = U[:, :p]        # (D, p)
    S_p = S[:p]            # (p,)
    Vh_p = Vh[:p, :]       # (p, c)

    # Edge weight matrix: W_edge = diag(S_p) @ Vh_p  (p, c)
    W_edge = (S_p.unsqueeze(1) * Vh_p)  # (p, c) — elementwise broadcast

    # Tail energy ratio
    tail_energy = S[p:].pow(2).sum() if p < len(S) else torch.tensor(0.0)
    tail_energy_ratio = (tail_energy / S.pow(2).sum()).item()

    return SpectralDecomposition(
        U_p=U_p.to(torch.float32),
        W_edge=W_edge.to(torch.float32),
        singular_values=S.to(torch.float32),
        rank_p=p,
        tail_energy_ratio=tail_energy_ratio,
        W_ridge=W_ridge.to(torch.float32),
    )


def edge_inference(
    z: torch.Tensor,
    W_edge: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Simulate edge device inference: logits = W_edge^T z + bias.

    This is the only computation the extreme edge device performs.
    Complexity: O(p * c) MACs.

    Parameters
    ----------
    z : torch.Tensor
        Projected feature vector of shape (p,) or batch (n, p).
    W_edge : torch.Tensor
        Edge weight matrix of shape (p, c).
    bias : torch.Tensor
        Bias vector (teacher logit mean) of shape (c,).

    Returns
    -------
    logits : torch.Tensor
        Logit vector of shape (c,) or batch (n, c).
    """
    return z @ W_edge + bias  # (n, c) or (c,)
