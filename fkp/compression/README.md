# `compression/` — Regression, SVD, and Compressibility Diagnostic

This module is the algorithmic heart of FKP. It has three responsibilities:

| File | Purpose |
|---|---|
| `ridge.py` | Closed-form ridge regression (primal and dual forms) |
| `svd.py` | Spectral compression via truncated SVD |
| `pcls.py` | PCLS: the go/no-go linearity diagnostic |

---

## `ridge.py` — Closed-Form Ridge Regression

FKP fits a linear map `W: R^D → R^c` from ZCA-whitened features to centred
teacher logits using closed-form ridge regression.

### Why two forms?

The matrix inversion lemma gives two algebraically equivalent solutions:

| Form | When preferred | Cost |
|---|---|---|
| **Primal** `(E^T E + αI_D)^{-1} E^T H` | D ≤ m | O(D³) |
| **Dual** `E^T (E E^T + αI_m)^{-1} H` | D > m | O(m³) |

`ridge_auto()` selects the cheaper form automatically. This matters critically:
for ViT-B/16 multi-layer features, D can be 3072 with m=500, so the dual form
is ~37× faster than the primal.

```python
from fkp.compression.ridge import ridge_auto, ridge_primal, ridge_dual

W = ridge_auto(E_tilde, H_c, alpha=1.0)   # auto-selects based on D vs m
```

**Math test**: `tests/test_ridge_dual.py` verifies `ridge_primal ≡ ridge_dual`
within `1e-5` at D=2000, m=500 for α ∈ {0.1, 1.0, 10.0}.

---

## `svd.py` — Spectral Compression

After ridge regression, `W_ridge ∈ R^{D×c}` is factored via SVD:

```
W_ridge = U Σ V^T
```

FKP keeps only the top-p singular triplets (rank-p truncation), where p is
chosen so that the discarded tail energy is ≤ η (default 5%):

```
sum_{i=p+1}^{r} σ_i² / sum_{i=1}^{r} σ_i² ≤ η
```

This yields the split:
- **Gateway sends**: `z = U_p^T @ tilde_E  ∈ R^p`
- **Edge computes**: `logits = W_edge^T @ z + H_bar`, where `W_edge = Σ_p V_p`

```python
from fkp.compression.svd import spectral_compress

decomp = spectral_compress(W_ridge, eta_tail=0.05)
# decomp.U_p      — (D, p) gateway projection
# decomp.W_edge   — (p, c) edge weight matrix
# decomp.rank_p   — selected rank
```

---

## `pcls.py` — PCLS Diagnostic

PCLS (Pre-Compression Linearity Score) answers: *Is this teacher's feature space
linear enough for FKP to work?*

It is computed **before** fitting anything on the test set:

1. Split the calibration set into train (80%) and val (20%).
2. Fit `W_ridge` on the train split.
3. Report R² on the val split.

```
PCLS ≥ 0.8   →   HIGH compressibility — safe to proceed
0.7 ≤ PCLS < 0.8  →   MARGINAL — monitor agreement rate
PCLS < 0.7   →   LOW — switch teacher or add layers
```

```python
from fkp.compression.pcls import compute_pcls, PCLS_HIGH_THRESHOLD

result = compute_pcls(E_tilde, H_raw, alpha=1.0)
print(result)
# PCLS = 0.8842  (HIGH — safe to proceed.)
```
