"""
Generate Paper Figures.

Replicates the main figures from the FKP paper:
    - Figure 2: PCLS scatter plot (PCLS vs accuracy retention)
    - Figure 3: Certificate tightness (theoretical bound vs empirical error)
    - Figure 4: Edge model size comparison across methods

Usage:
    python scripts/generate_figures.py \
        --results_dir outputs/ \
        --output_dir figures/

Requirements:
    pip install matplotlib seaborn
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fkp.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FKP paper figures")
    parser.add_argument("--results_dir", type=str, default="outputs/",
                        help="Directory containing run subdirectories with metadata.json")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Output directory for figure files")
    parser.add_argument("--format", type=str, default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output figure format")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    return parser.parse_args()


def load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load metadata.json from all run directories."""
    results = []
    for metadata_file in results_dir.glob("*/metadata.json"):
        with open(metadata_file) as f:
            data = json.load(f)
        data["run_dir"] = str(metadata_file.parent.name)
        results.append(data)
    logger.info("Loaded %d experiment results from %s", len(results), results_dir)
    return results


def plot_pcls_scatter(results: list[dict], output_path: Path, fmt: str, dpi: int) -> None:
    """Figure 2: PCLS vs accuracy retention scatter plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping PCLS scatter plot.")
        return

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(7, 5))

    for r in results:
        pcls = r.get("pcls", None)
        agreement = r.get("agreement_rate", None)
        if pcls is None or agreement is None:
            continue
        teacher = r.get("teacher", "?")
        dataset = r.get("dataset", "?")
        ax.scatter(pcls, agreement * 100, s=80, zorder=5,
                   label=f"{teacher}/{dataset}")

    # Threshold lines
    ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1.5,
               label="PCLS = 0.8 threshold")
    ax.axhline(y=95.0, color="gray", linestyle=":", linewidth=1.2,
               label="95% accuracy retention")

    ax.set_xlabel("PCLS (Pre-Compression Linearity Score)", fontsize=13)
    ax.set_ylabel("Accuracy Retention (%)", fontsize=13)
    ax.set_title("PCLS predicts compression feasibility", fontsize=14)
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="lower right")

    out = output_path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PCLS scatter plot: %s", out)


def plot_certificate_tightness(results: list[dict], output_path: Path, fmt: str, dpi: int) -> None:
    """Figure 3: Theoretical bound vs empirical error as rank p varies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed. Skipping certificate tightness plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        cert = r.get("certificate", {})
        eps_delta = cert.get("eps_delta")
        emp_err = cert.get("empirical_error")
        if eps_delta is None or emp_err is None:
            continue
        rank_p = r.get("rank_p", 0)
        ax.scatter(rank_p, eps_delta, marker="^", color="steelblue", s=80,
                   label="Theoretical bound" if rank_p == results[0].get("rank_p") else "")
        ax.scatter(rank_p, emp_err, marker="o", color="darkorange", s=80,
                   label="Empirical error" if rank_p == results[0].get("rank_p") else "")

    ax.set_xlabel("Projection rank $p$", fontsize=13)
    ax.set_ylabel("Logit approximation error", fontsize=13)
    ax.set_title("Certificate tightness: bound vs empirical error", fontsize=14)
    ax.legend(fontsize=11)

    out = output_path.with_name("certificate_tightness").with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved certificate tightness plot: %s", out)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        logger.warning("No results found in %s. Run run_calibration.py first.", results_dir)
        return

    plot_pcls_scatter(results, output_dir / "fig_pcls_scatter", args.format, args.dpi)
    plot_certificate_tightness(results, output_dir / "fig_certificate", args.format, args.dpi)
    logger.info("All figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
