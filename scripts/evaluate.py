"""
FKP Evaluation Script.

Evaluates a calibrated FKP model on a test set, reporting:
    - Teacher accuracy (top-1, top-5)
    - FKP edge model accuracy (top-1, top-5)
    - Agreement rate (decision preservation)
    - Payload and latency statistics
    - Certificate bound ratio (theoretical / empirical)

Usage:
    python scripts/evaluate.py \
        --weights_dir outputs/plantvillage_resnet50 \
        --teacher resnet50 \
        --dataset plantvillage
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fkp.utils.seeding import seed_everything
from fkp.utils.logging import get_logger
from fkp.utils.metrics import (
    compute_accuracy,
    compute_topk_accuracy,
    agreement_rate,
    edge_model_size_kb,
    payload_bytes,
    macs_edge_inference,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FKP Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--weights_dir", type=str, required=True,
                        help="Directory with U_p.npy, W_edge.npy, bias.npy, metadata.json")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Teacher model name (overrides metadata)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (overrides metadata)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_weights(weights_dir: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    U_p = torch.from_numpy(np.load(weights_dir / "U_p.npy"))
    W_edge = torch.from_numpy(np.load(weights_dir / "W_edge.npy"))
    bias = torch.from_numpy(np.load(weights_dir / "bias.npy"))
    metadata_path = weights_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    return U_p, W_edge, bias, metadata


def evaluate(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    weights_dir = Path(args.weights_dir)

    logger.info("Loading FKP weights from %s", weights_dir)
    U_p, W_edge, bias, meta = load_weights(weights_dir)

    p, c = W_edge.shape
    logger.info("Edge model: rank_p=%d, n_classes=%d", p, c)
    logger.info("Edge model size : %.2f KB", edge_model_size_kb(p, c))
    logger.info("Payload         : %d bytes", payload_bytes(p))
    logger.info("MACs per infer  : %d", macs_edge_inference(p, c))

    if meta:
        logger.info("PCLS from calibration : %.4f", meta.get("pcls", float("nan")))
        cert = meta.get("certificate", {})
        if cert:
            logger.info("Certificate eps_delta : %.6f", cert.get("eps_delta", float("nan")))
            status = "PASSED ✓" if cert.get("decision_preserved") else "FAILED ✗"
            logger.info("Certificate status    : %s", status)

    # TODO: Implement actual dataset loading and full evaluation loop.
    # The evaluation loop should:
    #   1. Load the test set dataloader.
    #   2. Extract multi-layer features using HookManager.
    #   3. Apply the saved ZCA whitening statistics (load from weights_dir).
    #   4. Compute z = U_p^T @ tilde_E for each test sample.
    #   5. Compute logits_edge = z @ W_edge + bias.
    #   6. Compute teacher logits directly from the model.
    #   7. Report top-1 accuracy, top-5 accuracy, and agreement rate.
    logger.info(
        "Full evaluation requires dataset loading. "
        "Implement get_test_dataloader() for your specific dataset."
    )


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
