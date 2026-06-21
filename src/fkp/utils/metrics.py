"""
Evaluation metrics for FKP compression quality.

Covers classification accuracy, agreement rate between teacher and edge
model predictions, and communication payload estimation.
"""

from __future__ import annotations

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 classification accuracy.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logit matrix of shape (n, c).
    labels : torch.Tensor
        Ground-truth class indices of shape (n,).

    Returns
    -------
    accuracy : float
        Top-1 accuracy in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def compute_topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Compute top-k classification accuracy.

    Parameters
    ----------
    logits : torch.Tensor
        Predicted logit matrix of shape (n, c).
    labels : torch.Tensor
        Ground-truth class indices of shape (n,).
    k : int
        Number of top predictions to consider.

    Returns
    -------
    accuracy : float
        Top-k accuracy in [0, 1].
    """
    topk_preds = logits.topk(k, dim=1).indices    # (n, k)
    correct = topk_preds.eq(labels.unsqueeze(1))  # (n, k)
    return correct.any(dim=1).float().mean().item()


def agreement_rate(logits_teacher: torch.Tensor, logits_edge: torch.Tensor) -> float:
    """Fraction of samples where teacher and edge model predict the same class.

    This is the primary measure of decision preservation — the metric
    directly verified by Corollary 1 of the certificate.

    Parameters
    ----------
    logits_teacher : torch.Tensor
        Teacher logits of shape (n, c).
    logits_edge : torch.Tensor
        Edge model logits of shape (n, c).

    Returns
    -------
    rate : float
        Agreement rate in [0, 1].  1.0 = perfect teacher fidelity.
    """
    pred_teacher = logits_teacher.argmax(dim=1)
    pred_edge = logits_edge.argmax(dim=1)
    return (pred_teacher == pred_edge).float().mean().item()


def compression_ratio(teacher_params: int, edge_params: int) -> float:
    """Compute the compression ratio (teacher / edge).

    Parameters
    ----------
    teacher_params : int
        Total number of parameters in the teacher model.
    edge_params : int
        Total number of parameters in the edge model (W_edge + bias).

    Returns
    -------
    ratio : float
        Compression ratio.  E.g. 1000 means the edge model is 1000x smaller.
    """
    if edge_params == 0:
        raise ValueError("edge_params must be > 0")
    return teacher_params / edge_params


def payload_bytes(rank_p: int, dtype_bytes: int = 4) -> int:
    """Compute the communication payload size in bytes.

    The gateway transmits a p-dimensional float32 vector z = U_p^T @ tilde_E.

    Parameters
    ----------
    rank_p : int
        Projection rank p (number of floats transmitted).
    dtype_bytes : int
        Bytes per scalar.  Default: 4 (float32).

    Returns
    -------
    n_bytes : int
        Payload size in bytes.
    """
    return rank_p * dtype_bytes


def edge_model_size_kb(
    rank_p: int,
    n_classes: int,
    dtype_bytes: int = 4,
) -> float:
    """Compute the edge model storage footprint in kilobytes.

    Edge model stores W_edge (p x c) + bias b (c,).

    Parameters
    ----------
    rank_p : int
        Projection rank p.
    n_classes : int
        Number of output classes c.
    dtype_bytes : int
        Bytes per scalar.  Default: 4 (float32).

    Returns
    -------
    size_kb : float
        Edge model size in kilobytes.
    """
    n_params = rank_p * n_classes + n_classes   # W_edge + b
    return (n_params * dtype_bytes) / 1024.0


def macs_edge_inference(rank_p: int, n_classes: int) -> int:
    """Compute multiply-accumulate operations for one edge inference.

    The edge performs: logit = W_edge^T @ z + b
      - W_edge^T @ z  : p * c MACs
      - + b           : c additions (negligible)

    Parameters
    ----------
    rank_p : int
        Projection rank p.
    n_classes : int
        Number of output classes c.

    Returns
    -------
    macs : int
        Number of MACs.
    """
    return rank_p * n_classes
