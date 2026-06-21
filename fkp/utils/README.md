# `utils/` — Seeding, Metrics, and Logging

Lightweight utilities used across all FKP modules.

| File | Purpose |
|---|---|
| `seeding.py` | `seed_everything()` — fully deterministic execution |
| `metrics.py` | accuracy, agreement_rate, edge model size, MACs |
| `logging.py` | Structured logger factory |

---

## `seeding.py`

**Every experiment script must call `seed_everything()` as its first line.**

```python
from fkp.utils.seeding import seed_everything
seed_everything(42)   # covers Python random, NumPy, PyTorch CPU+CUDA
```

Sets: `random`, `PYTHONHASHSEED`, `numpy.random`, `torch.manual_seed`,
`torch.cuda.manual_seed_all`, `CUBLAS_WORKSPACE_CONFIG`, and
`torch.use_deterministic_algorithms(True)`.

All paper experiments use **seed=42**.

---

## `metrics.py`

```python
from fkp.utils.metrics import (
    compute_accuracy,       # top-1 accuracy
    compute_topk_accuracy,  # top-k accuracy
    agreement_rate,         # teacher/edge prediction agreement
    edge_model_size_kb,     # (p * c + c) * 4 / 1024
    payload_bytes,          # p * 4  (float32 z vector)
    macs_edge_inference,    # p * c multiply-accumulates
)
```

### Decision preservation metric

`agreement_rate(logits_teacher, logits_edge)` is the primary empirical check
for the certificate's Corollary 1. It should be >= 0.95 for PCLS >= 0.8 pairs.

---

## `logging.py`

```python
from fkp.utils.logging import get_logger
logger = get_logger(__name__)
logger.info("Stage 2 complete: W_ridge shape=%s", tuple(W.shape))
```

Format: `2027-01-15 09:42:11 | INFO     | fkp.compression.ridge | ...`
