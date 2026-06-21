"""
Logit and feature centering utilities.

All centering is performed on the calibration set and the computed
statistics are stored for consistent application at inference time.
"""

from __future__ import annotations

import torch


def center_logits(H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center the teacher logit matrix by subtracting the column-wise mean.

    Parameters
    ----------
    H : torch.Tensor
        Teacher logit matrix of shape (m, c), where m is the number of
        calibration samples and c is the number of output classes.

    Returns
    -------
    H_c : torch.Tensor
        Centered logit matrix of shape (m, c).  H_c = H - 1 * H_bar^T.
    H_bar : torch.Tensor
        Column-wise mean vector of shape (c,).  This equals the bias term
        b in the edge classifier: b := H_bar.
    """
    if H.ndim != 2:
        raise ValueError(f"H must be 2-D (m, c), got shape {tuple(H.shape)}")

    H_bar = H.mean(dim=0)          # (c,)
    H_c = H - H_bar.unsqueeze(0)   # (m, c)
    return H_c, H_bar


def center_features(E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Center the feature design matrix by subtracting the sample-wise mean.

    Parameters
    ----------
    E : torch.Tensor
        Feature design matrix of shape (m, D).

    Returns
    -------
    E_c : torch.Tensor
        Centered feature matrix of shape (m, D).
    E_bar : torch.Tensor
        Column-wise mean vector of shape (D,).
    """
    if E.ndim != 2:
        raise ValueError(f"E must be 2-D (m, D), got shape {tuple(E.shape)}")

    E_bar = E.mean(dim=0)          # (D,)
    E_c = E - E_bar.unsqueeze(0)   # (m, D)
    return E_c, E_bar
