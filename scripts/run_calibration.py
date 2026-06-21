"""
FKP Calibration Pipeline — Full CLI Entry Point.

Runs the complete 4-stage FKP pipeline on a given teacher + dataset:
    Stage 1: Feature extraction & PCLS diagnostic
    Stage 2: ZCA whitening + Ridge Regression
    Stage 3: Spectral compression (SVD)
    Stage 4: Certificate computation & margin check

Outputs:
    outputs/<run_id>/
        U_p.npy           — gateway projection matrix
        W_edge.npy        — edge weight matrix
        bias.npy          — bias vector (H_bar)
        metadata.json     — run metadata, PCLS, certificate result
        certificate.txt   — human-readable certificate badge

Usage:
    python scripts/run_calibration.py \
        --config configs/experiments/main_table.yaml \
        --teacher resnet50 \
        --dataset plantvillage \
        --output_dir outputs/plantvillage_resnet50

    # Export to C after calibration:
    python edge_deployment/export_to_c.py \
        --weights_dir outputs/plantvillage_resnet50 \
        --output edge_deployment/include/fkp_inference.h
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from fkp.utils.seeding import seed_everything
from fkp.utils.logging import get_logger
from fkp.conditioning.zca import ZCAWhitener
from fkp.conditioning.centering import center_logits
from fkp.compression.pcls import compute_pcls
from fkp.compression.ridge import ridge_auto
from fkp.compression.svd import spectral_compress
from fkp.theory.certificate import FKPCertificate
from fkp.utils.metrics import (
    edge_model_size_kb,
    payload_bytes,
    macs_edge_inference,
    agreement_rate,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FKP Calibration Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teacher", type=str, default="resnet50",
                        help="Teacher model name (timm or 'clip', 'dinov2_small')")
    parser.add_argument("--dataset", type=str, default="plantvillage",
                        help="Dataset name (plantvillage, cifar100, domainnet)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML experiment config (overrides CLI args)")
    parser.add_argument("--output_dir", type=str, default="outputs/run",
                        help="Output directory for weights and metadata")
    parser.add_argument("--calib_size", type=int, default=500,
                        help="Calibration set size m")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regularization penalty")
    parser.add_argument("--delta", type=float, default=0.01,
                        help="Certificate failure probability")
    parser.add_argument("--eta_tail", type=float, default=0.05,
                        help="Spectral tail energy threshold for rank selection")
    parser.add_argument("--rank_p", type=int, default=None,
                        help="Force projection rank p (auto-selected if None)")
    parser.add_argument("--lambda_zca", type=float, default=1e-4,
                        help="ZCA whitening regularization")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Train/val split ratio for PCLS")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for feature extraction (cpu/cuda)")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML experiment config."""
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")


def run_pipeline(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  FKP Calibration Pipeline")
    logger.info("  Teacher  : %s", args.teacher)
    logger.info("  Dataset  : %s", args.dataset)
    logger.info("  Calib m  : %d", args.calib_size)
    logger.info("  alpha    : %.4f", args.alpha)
    logger.info("  delta    : %.4f", args.delta)
    logger.info("=" * 60)

    # ── Stage 0: Load teacher and calibration data ────────────────────────
    logger.info("[Stage 0] Loading teacher model and calibration data...")
    try:
        from fkp.models.zoo import load_teacher
        teacher, layer_names = load_teacher(args.teacher, device=args.device)
        logger.info("Teacher layers: %s", layer_names)
    except (ImportError, Exception) as e:
        logger.error("Could not load teacher '%s': %s", args.teacher, e)
        logger.info("Falling back to synthetic calibration data for demo.")
        teacher = None
        layer_names = []

    # For demo / CI: generate synthetic features if no model available
    if teacher is None:
        logger.warning("Using synthetic data — install timm/transformers for real features.")
        m, D_per_layer, c = args.calib_size, 256, 38
        n_layers = 4
        layer_features = [torch.randn(m, D_per_layer) for _ in range(n_layers)]
        H_raw = torch.randn(m, c)
    else:
        raise NotImplementedError(
            "Real data loading not yet implemented in this demo script. "
            "Implement get_calibration_dataloader() for your dataset."
        )

    total_time = time.time()

    # ── Stage 1: PCLS diagnostic ──────────────────────────────────────────
    logger.info("[Stage 1] Computing PCLS diagnostic...")
    t0 = time.time()

    whitener_pcls = ZCAWhitener(lambda_zca=args.lambda_zca)
    E_tilde = whitener_pcls.fit_transform(layer_features, calibration_size=len(layer_features[0]))
    logger.info(whitener_pcls.summary())

    pcls_result = compute_pcls(
        E_tilde, H_raw,
        alpha=args.alpha,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    logger.info("PCLS: %s", pcls_result)

    if pcls_result.score < 0.7:
        logger.warning(
            "PCLS=%.4f < 0.7 — Low compressibility. "
            "Consider increasing m, adding layers, or changing teacher.",
            pcls_result.score,
        )

    t_pcls = time.time() - t0
    logger.info("Stage 1 complete in %.2fs", t_pcls)

    # ── Stage 2: ZCA whitening + Ridge Regression ─────────────────────────
    logger.info("[Stage 2] ZCA whitening + Ridge Regression...")
    t0 = time.time()

    H_c, H_bar = center_logits(H_raw)
    W_ridge = ridge_auto(E_tilde, H_c, alpha=args.alpha)

    t_ridge = time.time() - t0
    logger.info("Ridge complete: W_ridge shape=%s in %.2fs", tuple(W_ridge.shape), t_ridge)

    # ── Stage 3: Spectral compression ────────────────────────────────────
    logger.info("[Stage 3] Spectral compression (SVD)...")
    t0 = time.time()

    decomp = spectral_compress(W_ridge, eta_tail=args.eta_tail, rank_p=args.rank_p)
    logger.info(
        "Selected rank p=%d  |  tail energy ratio=%.4f",
        decomp.rank_p, decomp.tail_energy_ratio,
    )

    # Edge model statistics
    m_total = layer_features[0].shape[0]
    c = H_raw.shape[1]
    size_kb = edge_model_size_kb(decomp.rank_p, c)
    payload = payload_bytes(decomp.rank_p)
    macs = macs_edge_inference(decomp.rank_p, c)

    logger.info("Edge model size : %.2f KB", size_kb)
    logger.info("Payload         : %d bytes", payload)
    logger.info("MACs per infer  : %d", macs)

    t_svd = time.time() - t0
    logger.info("Stage 3 complete in %.2fs", t_svd)

    # ── Stage 4: Certificate ──────────────────────────────────────────────
    logger.info("[Stage 4] Computing theoretical certificate...")
    t0 = time.time()

    # Edge logits on calibration set (for empirical error measurement)
    z_calib = E_tilde @ decomp.U_p              # (m, p)
    logits_edge = z_calib @ decomp.W_edge + H_bar  # (m, c)

    cert_engine = FKPCertificate(alpha=args.alpha, delta=args.delta)
    cert_result = cert_engine.compute(
        E_tilde=E_tilde,
        H_c=H_c,
        H_raw=H_raw,
        W_ridge=W_ridge,
        singular_values=decomp.singular_values,
        rank_p=decomp.rank_p,
        logits_edge=logits_edge,
    )

    agreement = agreement_rate(H_raw, logits_edge)
    logger.info("Agreement rate (calib) : %.4f", agreement)

    print("\n" + str(cert_result))

    t_cert = time.time() - t0
    t_total = time.time() - total_time
    logger.info("Stage 4 complete in %.2fs  |  Total: %.2fs", t_cert, t_total)

    # ── Save outputs ──────────────────────────────────────────────────────
    logger.info("[Save] Writing outputs to %s", output_dir)
    import numpy as np

    np.save(output_dir / "U_p.npy", decomp.U_p.numpy())
    np.save(output_dir / "W_edge.npy", decomp.W_edge.numpy())
    np.save(output_dir / "bias.npy", H_bar.numpy())

    metadata = {
        "teacher": args.teacher,
        "dataset": args.dataset,
        "calib_size": m_total,
        "n_classes": c,
        "rank_p": decomp.rank_p,
        "tail_energy_ratio": decomp.tail_energy_ratio,
        "pcls": pcls_result.score,
        "edge_model_kb": size_kb,
        "payload_bytes": payload,
        "macs": macs,
        "certificate": {
            "eps_delta": cert_result.eps_delta,
            "eps_stab": cert_result.eps_stab,
            "eps_proj": cert_result.eps_proj,
            "gamma_min": cert_result.gamma_min,
            "decision_preserved": cert_result.decision_preserved,
            "empirical_error": cert_result.empirical_error,
            "bound_ratio": cert_result.bound_ratio,
        },
        "agreement_rate": agreement,
        "calibration_time_s": t_total,
        "alpha": args.alpha,
        "delta": args.delta,
        "seed": args.seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    cert_str = str(cert_result)
    with open(output_dir / "certificate.txt", "w") as f:
        f.write(cert_str + "\n")

    logger.info("All outputs saved to %s", output_dir)
    logger.info("To export C header: python edge_deployment/export_to_c.py "
                "--weights_dir %s", output_dir)


if __name__ == "__main__":
    args = parse_args()
    if args.config:
        cfg = load_config(args.config)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    run_pipeline(args)
