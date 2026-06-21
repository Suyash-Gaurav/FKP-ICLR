# `scripts/` — CLI Entry Points

Three command-line scripts covering the full FKP workflow.

| Script | Purpose |
|---|---|
| `run_calibration.py` | Full 4-stage pipeline: PCLS → ZCA+Ridge → SVD → Certificate |
| `evaluate.py` | Accuracy + agreement rate on a test set |
| `generate_figures.py` | Reproduce paper figures from `outputs/` |

---

## `run_calibration.py`

The main script. Runs all four FKP stages and saves:

```
outputs/<run>/
├── U_p.npy           gateway projection matrix (D, p)
├── W_edge.npy        edge weight matrix (p, c)
├── bias.npy          bias vector H_bar (c,)
├── metadata.json     all metrics, PCLS, certificate result
└── certificate.txt   human-readable certificate badge
```

### Full usage

```bash
python scripts/run_calibration.py \
    --teacher resnet50 \          # timm model name
    --dataset plantvillage \      # see configs/datasets/
    --config configs/experiments/main_table.yaml \
    --calib_size 500 \            # calibration set size m
    --alpha 1.0 \                 # ridge regularization
    --delta 0.01 \                # certificate failure probability
    --eta_tail 0.05 \             # spectral tail threshold for rank selection
    --lambda_zca 1e-4 \           # ZCA whitening regularization
    --seed 42 \
    --output_dir outputs/plantvillage_resnet50
```

### Config override

If `--config` is provided, the YAML values override the corresponding CLI
arguments. This allows fully reproducible runs with a single flag:

```bash
python scripts/run_calibration.py --config configs/experiments/main_table.yaml
```

---

## `evaluate.py`

Loads saved FKP weights and evaluates on a held-out test set:

```bash
python scripts/evaluate.py \
    --weights_dir outputs/plantvillage_resnet50 \
    --teacher resnet50 \
    --dataset plantvillage
```

Reports: top-1 accuracy, top-5 accuracy, agreement rate, edge model size, payload bytes, MACs.

---

## `generate_figures.py`

Generates PDF/PNG figures from all `metadata.json` files in `outputs/`:

```bash
python scripts/generate_figures.py \
    --results_dir outputs/ \
    --output_dir figures/ \
    --format pdf          # pdf | png | svg
```

Requires `matplotlib` and `seaborn` (`pip install fkp[viz]`).

---

## Reproducibility shortcut

```bash
make reproduce-main-table    # runs all teacher × dataset combinations
make reproduce-ablation      # runs ablation study
```
