# `data/` — Dataset Storage

This directory holds raw and preprocessed datasets.
**It is gitignored** — datasets are not committed to the repository.

## Download

```bash
bash data/download_datasets.sh
```

This script auto-downloads:
- **CIFAR-10** and **CIFAR-100** via torchvision (auto)
- **PlantVillage** via HuggingFace `datasets` (auto)
- **Cassava** via HuggingFace `datasets` (auto)
- **DomainNet**: manual steps required (see script output)

## Expected layout after download

```
data/
├── cifar10/
│   └── cifar-10-batches-py/
├── cifar100/
│   └── cifar-100-python/
├── plantvillage/
│   └── jkang37___plant-village/    (HuggingFace cache)
├── cassava/
│   └── cassava/
└── domainnet/
    ├── real/
    │   └── <345 class folders>
    └── sketch/
        └── <345 class folders>
```

## Disk space requirements

| Dataset | Size |
|---|---|
| CIFAR-10 | ~170 MB |
| CIFAR-100 | ~170 MB |
| PlantVillage | ~2.2 GB |
| Cassava | ~2.7 GB |
| DomainNet (real + sketch) | ~10 GB |

## Calibration sampling

Only `calib_size=500` samples are used from each dataset for FKP calibration.
The sampling is always done with `seed=42` for reproducibility (see `seeding.py`).
The remaining samples are used exclusively for test-set evaluation.
