# `theory/` — Finite-Sample Bounds and Decision-Preservation Certificate

This module implements the theoretical core of the FKP paper: a finite-sample
certificate that guarantees the edge model preserves the teacher's classification
decision on unseen test samples with high probability.

| File | Purpose |
|---|---|
| `bounds.py` | Individual bound components: stability, feature norm, projection loss |
| `certificate.py` | `FKPCertificate`: assembles bounds into a printable certificate |

---

## The Certificate (Theorem 1 + Corollary 1)

The certificate answers: *Can we prove that the edge model will predict the same
class as the teacher on a new sample, with probability ≥ 1 - δ?*

The total approximation error is bounded as:

```
ε_δ = √ε_stab + √ε_proj
```

where:

| Component | Meaning |
|---|---|
| `ε_stab` | Ridge regression uniform-stability bound: how much W changes when one calibration sample is removed |
| `ε_proj` | Spectral tail energy: how much information is lost by the rank-p truncation |
| `R_δ` | Sub-Gaussian feature norm bound: worst-case feature norm with probability 1−δ |
| `γ_min` | Minimum margin of the teacher across calibration samples |

**Corollary 1 (Decision Preservation)**:
With probability ≥ 1 − δ over a new test sample, the edge model predicts the
same class as the teacher if:

```
ε_δ < γ_min / 2
```

---

## `bounds.py`

```python
from fkp.theory.bounds import stability_bound, feature_norm_bound, projection_loss_bound

# Uniform-stability bound
result = stability_bound(E_tilde, H_c, W_ridge, alpha=1.0, delta=0.01)
# result.eps_stab, result.E_reg, result.gamma, result.c_delta, result.R_delta

# Sub-Gaussian feature norm bound
R_delta = feature_norm_bound(E_tilde, delta=0.01)

# Spectral projection loss bound
eps_proj = projection_loss_bound(singular_values, rank_p=64, R_delta=R_delta)
```

---

## `certificate.py`

```python
from fkp.theory.certificate import FKPCertificate

cert = FKPCertificate(alpha=1.0, delta=0.01)
result = cert.compute(
    E_tilde=E_tilde,
    H_c=H_c,
    H_raw=H_raw,
    W_ridge=W_ridge,
    singular_values=decomp.singular_values,
    rank_p=decomp.rank_p,
)
print(result)
```

Output:

```
============================================================
  FKP Certificate  (delta=0.0100)
============================================================
  R_delta           = 12.3821
  eps_stab          = 0.041823
  eps_proj          = 0.003214
  eps_delta (total) = 0.261390
  gamma_min         = 3.646800  (gamma_min/2 = 1.823400)
  empirical error   = 0.043102
  bound ratio       = 6.063x
  CERTIFICATE PASSED ✓
============================================================
```

**Math test**: `tests/test_certificate.py` verifies:
- `ε_δ = √ε_stab + √ε_proj` to **1e-10** precision
- All components are non-negative and finite
- `projection_loss_bound(..., rank_p=r) == 0.0` at full rank

---

## Parameter guidance

| Parameter | Recommended | Effect |
|---|---|---|
| `alpha` | 1.0 (tune via PCLS) | Larger → tighter stability, looser fit |
| `delta` | 0.01 for paper results | Smaller → larger `R_δ` and `c_δ` |
| `eta_tail` | 0.05 | Controls rank p; smaller → larger p → smaller ε_proj |
