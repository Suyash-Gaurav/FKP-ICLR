"""
Pre-Compression Linearity Score (PCLS).

PCLS is an out-of-sample coefficient of determination (R^2) that quantifies
how well the teacher's decision boundary can be approximated by a linear
function of the conditioned multi-layer embedding.

    PCLS = 1 - ||H_val - H_hat_val||_F^2 / ||H_val - 1*H_bar_val^T||_F^2

Interpretation:
    PCLS >= 0.8  ->  High linear compressibility; FKP is expected to succeed.
    PCLS in [0.7, 0.8)  ->  Moderate; consider increasing p or m.
    PCLS < 0.7   ->  Low; the teacher's boundary is fundamentally non-linear
                     in feature space.  Consider a different architecture.

Reference:
    §3.2 "Pre-Compression Linearity Score (PCLS)" and Definition 3 in FKP.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fkp.compression.ridge import ridge_auto
from fkp.conditioning.centering import center_logits


PCLS_HIGH_THRESHOLD = 0.8
PCLS_MARGINAL_THRESHOLD = 0.7


@dataclass
class PCLSResult:
    """Result of the PCLS computation.

    Attributes
    ----------
    score : float
        The PCLS value in (-inf, 1].
    interpretation : str
        Human-readable interpretation of the score.
    train_size : int
        Number of calibration samples used for training.
    val_size : int
        Number of samples used for validation.
    """
    score: float
    interpretation: str
    train_size: int
    val_size: int

    def __str__(self) -> str:
        return (
            f"PCLS = {self.score:.4f}  ({self.interpretation})\n"
            f"  Train size: {self.train_size}, Val size: {self.val_size}"
        )


def compute_pcls(
    E_tilde: torch.Tensor,
    H: torch.Tensor,
    alpha: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> PCLSResult:
    """Compute the Pre-Compression Linearity Score (PCLS).

    Procedure (Definition 3 in the paper):
      1. Randomly partition calibration indices into 80% train / 20% val.
      2. Center logits on the training split.
      3. Fit ridge regression on the training split.
      4. Predict on the validation split.
      5. Compute R^2 between predicted and actual validation logits.

    Parameters
    ----------
    E_tilde : torch.Tensor
        Conditioned (ZCA-whitened) feature design matrix of shape (m, D).
    H : torch.Tensor
        Raw (un-centered) teacher logit matrix of shape (m, c).
    alpha : float
        Ridge regularization penalty.  Default: 1.0.
    train_ratio : float
        Fraction of calibration samples used for training.  Default: 0.8.
    seed : int
        Random seed for the train/val split (for reproducibility).

    Returns
    -------
    PCLSResult
        Dataclass with the PCLS score and metadata.
    """
    if E_tilde.ndim != 2 or H.ndim != 2:
        raise ValueError("E_tilde and H must be 2-D tensors.")
    if E_tilde.shape[0] != H.shape[0]:
        raise ValueError("E_tilde and H must have the same number of rows.")
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")

    m = E_tilde.shape[0]

    # --- Reproducible train/val split ---
    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(m, generator=generator)
    n_train = max(1, int(m * train_ratio))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    if len(val_idx) == 0:
        raise ValueError(
            f"Validation set is empty with m={m} and train_ratio={train_ratio}. "
            "Increase calibration set size or decrease train_ratio."
        )

    E_train = E_tilde[train_idx]   # (n_train, D)
    H_train = H[train_idx]         # (n_train, c)
    E_val = E_tilde[val_idx]       # (n_val, D)
    H_val = H[val_idx]             # (n_val, c)

    # Center logits on the training split
    H_train_c, H_bar_train = center_logits(H_train)   # (n_train, c), (c,)

    # Fit ridge on training split
    W_train = ridge_auto(E_train, H_train_c, alpha=alpha)   # (D, c)

    # Predict on validation: H_hat_val = E_val @ W_train + H_bar_train
    H_hat_val = E_val @ W_train + H_bar_train.unsqueeze(0)  # (n_val, c)

    # R^2 score (multi-output, Frobenius-norm version)
    H_bar_val = H_val.mean(dim=0, keepdim=True)              # (1, c)
    ss_res = (H_val - H_hat_val).pow(2).sum()
    ss_tot = (H_val - H_bar_val).pow(2).sum()

    if ss_tot.item() == 0.0:
        # All validation logits are identical — degenerate case
        score = 1.0 if ss_res.item() < 1e-12 else float("-inf")
    else:
        score = 1.0 - (ss_res / ss_tot).item()

    interpretation = _interpret_pcls(score)

    return PCLSResult(
        score=score,
        interpretation=interpretation,
        train_size=n_train,
        val_size=len(val_idx),
    )


def _interpret_pcls(score: float) -> str:
    """Return a human-readable interpretation of a PCLS score."""
    if score >= PCLS_HIGH_THRESHOLD:
        return "HIGH compressibility — FKP expected to succeed."
    if score >= PCLS_MARGINAL_THRESHOLD:
        return (
            "MARGINAL compressibility — consider increasing m, adding layers, "
            "or increasing projection rank p."
        )
    return (
        "LOW compressibility — teacher boundary is non-linear. "
        "Consider switching teacher architecture or adding more calibration data."
    )
