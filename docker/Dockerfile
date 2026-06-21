# ── FKP — Reproducibility Docker Image ───────────────────────────────────────
# Provides an exact environment for reproducing all paper results.
#
# Build:   docker build -t fkp-iclr2027:latest -f docker/Dockerfile .
# Run:     docker run --rm -it fkp-iclr2027:latest bash
# With GPU: docker run --rm -it --gpus all fkp-iclr2027:latest bash
#
# The container automatically runs `make test` on startup to verify
# the math unit tests pass in the packaged environment.

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

LABEL maintainer="Suyash Gaurav <s23224522@al.tiu.ac.jp>"
LABEL description="FKP: Feature Knowledge Projection — ICLR 2027 reproducibility image"

# ── System dependencies ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    gcc \
    make \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────
WORKDIR /workspace

# ── Copy and install FKP package ──────────────────────────────────────────
COPY . /workspace/FKP/
WORKDIR /workspace/FKP

# Install pinned dependencies for exact reproducibility
RUN pip install --no-cache-dir -r requirements.txt

# Install the FKP package in editable mode
RUN pip install --no-cache-dir -e ".[models]"

# ── Verify math unit tests pass ───────────────────────────────────────────
RUN python -m pytest tests/test_ridge_dual.py tests/test_zca.py tests/test_certificate.py \
    -v --no-header -q && echo "All math tests passed."

# ── Default command ────────────────────────────────────────────────────────
CMD ["bash"]

# To run the full calibration pipeline:
#   docker run --rm fkp-iclr2027:latest \
#     python scripts/run_calibration.py --teacher resnet50 --dataset cifar100
