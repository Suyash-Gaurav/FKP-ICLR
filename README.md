# Feature Knowledge Projection (FKP)

> **Label-Free, Certifiable Edge Deployment of Frozen Foundation Models**
> 
> Suyash Gaurav — Tokyo International University
> Preprint 


[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is FKP?

FKP is a **tuning-free** framework for deploying frozen foundation models (ViT, ResNet, CLIP, DINOv2) on extreme edge devices such as ESP32 / STM32 microcontrollers.

Instead of compressing weights, FKP compresses the **communication payload** of split-feature inference:

```
Gateway (Raspberry Pi / Jetson Nano)
  ├── Frozen teacher model  →  multi-layer features  →  ZCA whitening  →  U_p projection
  └── Transmits z ∈ R^p  (256 bytes for p=64, float32)
                         ↓  BLE / LoRa
Extreme Edge (ESP32 / STM32)
  └── logits = W_edge^T @ z + bias  (2,500 MACs, ~500 µs)
```

**Key properties:**
| Property | FKP | Knowledge Distillation | Post-Training Quantization |
|---|---|---|---|
| Labels required | ❌ None | ✅ Required | ❌ None |
| Gradients required | ❌ None | ✅ Required | ❌ None |
| Fine-tuning required | ❌ None | ✅ Required | ❌ None |
| Finite-sample certificate | ✅ Yes | ❌ No | ❌ No |
| Split architecture | ✅ Yes | ❌ No | ❌ No |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/suyashgaurav/FKP-ICLR2027.git
cd FKP-ICLR2027
pip install -e ".[models,data]"
```

### 2. Run the calibration pipeline (demo mode)

```bash
python scripts/run_calibration.py \
    --teacher resnet50 \
    --dataset plantvillage \
    --calib_size 500 \
    --alpha 1.0 \
    --delta 0.01 \
    --output_dir outputs/plantvillage_resnet50
```

### 3. Export to C for MCU deployment

```bash
python edge_deployment/export_to_c.py \
    --weights_dir outputs/plantvillage_resnet50 \
    --output edge_deployment/include/fkp_inference.h
```

### 4. Run math tests

```bash
make test-math
```

---

## Repository Structure

```text
FKP-ICLR2027/
├── configs/                  # YAML configs for reproducibility          → configs/README.md
│   ├── datasets/             # plantvillage.yaml, cifar100.yaml, domainnet.yaml
│   ├── teachers/             # resnet50.yaml, vit_b16.yaml, clip.yaml
│   └── experiments/          # main_table.yaml, ablation_zca.yaml
│
├── data/                     # Gitignored; datasets downloaded here      → data/README.md
│   └── download_datasets.sh
│
├── src/fkp/                  # Core Python Package                       → src/fkp/README.md
│   ├── models/               # Teacher wrappers (hooks.py, zoo.py)       → src/fkp/models/README.md
│   ├── conditioning/         # ZCA whitening + logit centering           → src/fkp/conditioning/README.md
│   ├── compression/          # Ridge, SVD, PCLS                         → src/fkp/compression/README.md
│   ├── theory/               # Bounds + Certificate                      → src/fkp/theory/README.md
│   └── utils/                # Seeding, metrics, logging                 → src/fkp/utils/README.md
│
├── edge_deployment/          # Bare-metal C for ESP32 / STM32 / RP2040  → edge_deployment/README.md
│   ├── include/fkp_inference.h   # Auto-generated weight header
│   ├── src/main.c                # Bare-metal inference loop
│   ├── CMakeLists.txt
│   └── export_to_c.py            # Python → C array exporter
│
├── scripts/                  # CLI entry points                          → scripts/README.md
│   ├── run_calibration.py    # Full 4-stage pipeline
│   ├── evaluate.py           # Accuracy + agreement evaluation
│   └── generate_figures.py   # Paper figure generation
│
├── tests/                    # PyTest math verification suite            → tests/README.md
│   ├── test_ridge_dual.py    # Primal == Dual within 1e-5
│   ├── test_zca.py           # Empirical covariance == Identity within 1e-4
│   └── test_certificate.py   # Certificate components and decomposition
│
├── notebooks/                # Jupyter notebooks for paper figures       → notebooks/README.md
│   └── pcls_scatter.ipynb    # Figure 2: PCLS scatter plot
│
├── docker/                   # Reproducibility container                 → docker/README.md
│   └── Dockerfile
├── .github/workflows/ci.yml  # CI: math tests + code quality on every push
├── Makefile
├── pyproject.toml
└── requirements.txt
```

---

## Sub-folder Documentation

Every directory has its own `README.md` with detailed explanations:

| Directory | README | What it covers |
|---|---|---|
| `src/fkp/` | [src/fkp/README.md](src/fkp/README.md) | Package overview, import patterns, design rules |
| `src/fkp/compression/` | [compression/README.md](src/fkp/compression/README.md) | Ridge primal/dual, SVD rank selection, PCLS diagnostic |
| `src/fkp/conditioning/` | [conditioning/README.md](src/fkp/conditioning/README.md) | ZCA whitening (with PCA-ZCA hybrid), logit centering |
| `src/fkp/theory/` | [theory/README.md](src/fkp/theory/README.md) | Stability bounds, projection loss, certificate theorem |
| `src/fkp/models/` | [models/README.md](src/fkp/models/README.md) | HookManager, teacher model zoo, pooling strategies |
| `src/fkp/utils/` | [utils/README.md](src/fkp/utils/README.md) | `seed_everything()`, metrics, structured logging |
| `tests/` | [tests/README.md](tests/README.md) | Math test philosophy, what each test proves, tolerances |
| `edge_deployment/` | [edge_deployment/README.md](edge_deployment/README.md) | Export workflow, C build, MCU flash instructions |
| `scripts/` | [scripts/README.md](scripts/README.md) | CLI flags, config override, reproducibility shortcuts |
| `configs/` | [configs/README.md](configs/README.md) | YAML schema, adding datasets/teachers/experiments |
| `notebooks/` | [notebooks/README.md](notebooks/README.md) | Jupyter setup, figure notebook descriptions |
| `data/` | [data/README.md](data/README.md) | Download instructions, disk space, expected layout |
| `docker/` | [docker/README.md](docker/README.md) | Build, run, GPU support, reproducing tables inside Docker |

---

## The FKP Pipeline

### Stage 1: PCLS Diagnostic

```python
from fkp.compression.pcls import compute_pcls
result = compute_pcls(E_tilde, H_raw, alpha=1.0)
print(result)
# PCLS = 0.8842  (HIGH compressibility — FKP expected to succeed.)
```

PCLS ≥ 0.8 → safe to proceed. PCLS < 0.7 → switch architecture.

### Stage 2: ZCA Whitening + Ridge Regression

```python
from fkp.conditioning.zca import ZCAWhitener
from fkp.conditioning.centering import center_logits
from fkp.compression.ridge import ridge_auto

whitener = ZCAWhitener(lambda_zca=1e-4)
E_tilde = whitener.fit_transform(layer_features, calibration_size=m)

H_c, H_bar = center_logits(H_raw)
W_ridge = ridge_auto(E_tilde, H_c, alpha=1.0)   # primal if D≤m, dual if D>m
```

### Stage 3: Spectral Compression

```python
from fkp.compression.svd import spectral_compress
decomp = spectral_compress(W_ridge, eta_tail=0.05)
# decomp.U_p     — gateway projection  (D, p)
# decomp.W_edge  — edge weight matrix  (p, c)
# decomp.rank_p  — selected rank p
```

### Stage 4: Certificate

```python
from fkp.theory.certificate import FKPCertificate
cert = FKPCertificate(alpha=1.0, delta=0.01)
result = cert.compute(E_tilde, H_c, H_raw, W_ridge, decomp.singular_values, decomp.rank_p)
print(result)
# ============================================================
#   FKP Certificate  (delta=0.0100)
# ============================================================
#   eps_stab          = 0.041823
#   eps_proj          = 0.003214
#   eps_delta (total) = 0.261390
#   gamma_min / 2     = 1.823400
#   CERTIFICATE PASSED ✓
# ============================================================
```

---

## Reproducibility

### Reproduce Table 1 (main results)

```bash
make reproduce-main-table
```

### Reproduce Table 2 (ablation study)

```bash
make reproduce-ablation
```

### Docker (exact environment)

```bash
docker build -t fkp-iclr2027:latest -f docker/Dockerfile .
docker run --rm -it fkp-iclr2027:latest bash
```

---

## Code Quality Rules

All PRs are automatically checked by GitHub Actions:

1. **`make test-math`** — Primal/dual ridge equivalence, ZCA identity covariance, certificate non-negativity.
2. **`ruff check`** — Linting.
3. **`black --check`** — Formatting.
4. **`mypy`** — Type checking.

See the [intern onboarding guide](docs/intern_onboarding.md) for the phased implementation plan.

---

## Citation

```bibtex
@inproceedings{gaurav2027fkp,
  title     = {Feature Knowledge Projection: Label-Free, Certifiable Edge Deployment of Frozen Foundation Models},
  author    = {Gaurav, Suyash},
  year      = {2027},
  note      = {Preprint}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
