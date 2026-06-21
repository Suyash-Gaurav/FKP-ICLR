# `src/fkp/` — Core Python Package

This is the root of the `fkp` Python package. It exposes the four algorithmic modules
(compression, conditioning, theory, models) plus utilities.

## Package layout

```
src/fkp/
├── compression/     Ridge regression + SVD compression + PCLS diagnostic  →  compression/README.md
├── conditioning/    ZCA whitening + logit centering                        →  conditioning/README.md
├── theory/          Uniform-stability bounds + finite-sample certificate   →  theory/README.md
├── models/          Teacher wrappers and hook-based feature extractor      →  models/README.md
└── utils/           Seeding, metrics, structured logging                  →  utils/README.md
```

## Typical import pattern

```python
# Complete 4-stage pipeline
from fkp.conditioning.zca      import ZCAWhitener
from fkp.conditioning.centering import center_logits
from fkp.compression.pcls       import compute_pcls
from fkp.compression.ridge      import ridge_auto
from fkp.compression.svd        import spectral_compress
from fkp.theory.certificate     import FKPCertificate
from fkp.utils.seeding          import seed_everything
```

## Design rules

- **No sklearn.** All linear algebra uses `torch.linalg` directly.
- **No `torch.nn.Linear`** for the core math. Raw matrix operations only.
- **No side effects on import.** Importing any submodule must not download weights, allocate GPU memory, or write files.
- All public functions carry type annotations and NumPy-style docstrings.

## Version

See `pyproject.toml` at the repo root for the current version and changelog.
