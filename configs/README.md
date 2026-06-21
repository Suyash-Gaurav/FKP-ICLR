# `configs/` — Experiment Configuration Files

All experiment parameters are declared in YAML files here. The goal is that
any paper result can be reproduced by pointing a script at a single YAML file.

```
configs/
├── datasets/           Per-dataset settings
│   ├── plantvillage.yaml
│   ├── cifar100.yaml
│   └── domainnet.yaml
├── teachers/           Per-teacher model settings and default hyperparameters
│   ├── resnet50.yaml
│   ├── vit_b16.yaml
│   └── clip.yaml
└── experiments/        Cross-product (teacher × dataset) experiment definitions
    ├── main_table.yaml     Table 1 in paper
    └── ablation_zca.yaml   Table 2 (ablation study)
```

---

## `datasets/`

Each dataset YAML defines:
- `n_classes`, `image_size`, normalisation mean/std
- Download path or HuggingFace dataset ID
- FKP-specific settings: `calib_size` (m) and which split to sample from

### Adding a new dataset

1. Copy `configs/datasets/cifar100.yaml` and fill in the fields.
2. Implement `get_calibration_dataloader(name)` in `scripts/run_calibration.py`.
3. Run `bash data/download_datasets.sh` to download the data.

---

## `teachers/`

Each teacher YAML defines:
- `backend`: `timm` | `transformers`
- `model_id`: exact model ID string for the backend
- `hook_layers`: list of submodule names to hook for feature extraction
- `total_feature_dim`: sum of all layer dimensions (D)
- `default_alpha` and `default_rank_p`: paper-reported defaults

### Pooling strategy

- CNNs (ResNet, MobileNet): `pool: avg` (global average pool of spatial feature map)
- Transformers (ViT, CLIP, DINOv2): `pool: cls_token` (first token)

### Note on D > m

For ViT-B/16 (`total_feature_dim=3072`) with `calib_size=500`, D > m and the
PCA-ZCA hybrid in `ZCAWhitener` is triggered automatically. No manual
configuration required.

---

## `experiments/`

Each experiment YAML lists the full cross-product of teachers and datasets,
along with shared FKP hyperparameters. Pass with `--config`:

```bash
python scripts/run_calibration.py --config configs/experiments/main_table.yaml
```

### `main_table.yaml` — Table 1

5 teachers × 3 datasets = 15 runs.
Shared: `alpha=1.0`, `delta=0.01`, `eta_tail=0.05`, `calib_size=500`, `seed=42`.

### `ablation_zca.yaml` — Table 2

5 variants on ResNet-50 / CIFAR-100:
- Full FKP (baseline)
- Without ZCA
- Single-layer features
- Without SVD compression
- PCA projection instead of FKP SVD
