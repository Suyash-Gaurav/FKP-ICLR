# `tests/` — Mathematical Verification Suite

> "A unit test for FKP isn't 'does the code run without crashing.'
>  A unit test is 'does the empirical covariance of the ZCA-whitened
>  features equal the Identity matrix within 1e-4 tolerance?'"
>
> — Intern Onboarding Guide, Phase 2

---

## Philosophy

Every test in this suite verifies a **mathematical guarantee** stated in the
FKP paper, not merely runtime behaviour. If any of these tests fails, the
paper's claims cannot be reproduced.

## Test files

| File | What it proves | Tolerance |
|---|---|---|
| `test_ridge_dual.py` | Primal == Dual ridge at D=2000, m=500 | **1e-5** |
| `test_zca.py` | Empirical covariance of ZCA output == I | **1e-4** |
| `test_certificate.py` | ε_δ = √ε_stab + √ε_proj; all components ≥ 0 | **1e-10** |

---

## Running the tests

```bash
# Full suite with coverage
make test

# Math-only (fast, no model weights needed)
make test-math

# Single file
pytest tests/test_ridge_dual.py -v
```

---

## `test_ridge_dual.py` — Ridge Primal ≡ Dual

Verifies the matrix inversion lemma identity:

```
(E^T E + αI)^{-1} E^T H  ≡  E^T (E E^T + αI)^{-1} H
```

**Key test case**: D=2000, m=500, c=10, α ∈ {0.1, 1.0, 10.0}.
This is the canonical case from the intern onboarding guide where D > m,
so the dual form is computationally preferred.

Additional tests:
- D < m case
- `ridge_auto()` selects the correct form
- Noiseless recovery of true weights as α → 0
- Input validation (negative alpha, shape mismatch, 3-D inputs)

---

## `test_zca.py` — ZCA Identity Covariance

Verifies the core ZCA guarantee: `(1/m) Ẽ^T Ẽ ≈ I_D`.

Tests cover:
- Standard case (D < m): `max |C_emp - I| < 1e-4`
- PCA-ZCA hybrid (D > m): triggered when d_k > m for a layer
- Zero-mean property after whitening
- Multi-layer concatenation shape
- `transform()` consistency with `fit_transform()`
- Error handling (unfitted, layer count mismatch, negative lambda)

---

## `test_certificate.py` — Certificate Properties

Verifies the theoretical decomposition ε_δ = √ε_stab + √ε_proj:

- `|ε_δ - (√ε_stab + √ε_proj)| < 1e-10`
- All components are finite and non-negative
- `projection_loss_bound(..., rank_p=r) == 0.0` when using full rank
- Bound is monotone non-increasing in rank_p
- R_δ decreases with larger m and larger δ
- `FKPCertificate` rejects invalid `delta` and `alpha`
- End-to-end test on synthetic linear teacher data

---

## CI

These tests run automatically on every push and PR via
`.github/workflows/ci.yml` across Python 3.10, 3.11, and 3.12.
The CI also runs the key mathematical properties as inline Python one-liners
for maximum visibility in the GitHub Actions log.
