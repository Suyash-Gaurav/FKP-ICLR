# `conditioning/` — ZCA Whitening and Logit Centering

This module prepares raw features and teacher logits for the ridge regression step.
Poor conditioning is the single most common cause of high PCLS failure — always
apply ZCA whitening before any compression step.

| File | Purpose |
|---|---|
| `zca.py` | Multi-layer ZCA whitening with PCA-ZCA hybrid |
| `centering.py` | Teacher logit centering (H_bar extraction) |

---

## `zca.py` — ZCA Whitening

ZCA (Zero-phase Component Analysis) whitening transforms layer-k features
`φ_k ∈ R^{d_k}` so that the empirical covariance of the concatenated whitened
features equals `I_D`:

```
(1/m) * Ẽ^T Ẽ ≈ I_D
```

This is the key conditioning step that makes ridge regression well-posed even
for heterogeneous multi-layer feature concatenations.

### Per-layer computation

For each layer k with calibration matrix `Φ_k ∈ R^{m × d_k}`:

1. Centre: `Φ̄_k = (1/m) Σ φ_k^{(i)}`
2. Covariance: `C_k = (1/m) (Φ_k - Φ̄_k)^T (Φ_k - Φ̄_k) + λ I`
3. Eigen-decompose: `C_k = Q Λ Q^T`
4. Whitening matrix: `W_k = Q Λ^{-½} Q^T`

### PCA-ZCA hybrid (when `d_k > m`)

For ViT-B/16, a single layer can have `d_k=768 > m=500`, making the covariance
matrix rank-deficient. In this case the module automatically falls back to a
PCA-ZCA hybrid:

1. Compute SVD of the centred data matrix: `Φ̄_k = U S V^T`
2. Retain components that explain `variance_threshold` of total variance
3. Project to the reduced subspace, then apply ZCA there

```python
from fkp.conditioning.zca import ZCAWhitener

whitener = ZCAWhitener(lambda_zca=1e-4, pca_variance_threshold=0.99)
E_tilde = whitener.fit_transform(layer_features, calibration_size=m)
print(whitener.summary())
```

**Math test**: `tests/test_zca.py` verifies:
- `(1/m) Ẽ^T Ẽ ≈ I` within **1e-4**
- Zero mean after whitening (< 1e-6)
- PCA-ZCA triggered and correct when d > m

---

## `centering.py` — Teacher Logit Centering

Before ridge regression the teacher's logit vectors are centred:

```
H_bar = (1/m) Σ h^{(i)}    (mean teacher logit)
H_c   = H - H_bar           (centred logits)
```

This is essential because the edge model reconstructs `H_bar` as a fixed bias
vector, separating the cross-sample mean from the per-sample variation that
the ridge map captures.

```python
from fkp.conditioning.centering import center_logits

H_c, H_bar = center_logits(H_raw)   # H_raw: (m, c)
# H_bar is saved as bias.npy and loaded on the edge device
```
