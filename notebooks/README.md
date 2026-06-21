# `notebooks/` — Interactive Paper Figure Notebooks

Jupyter notebooks for generating and exploring the paper's figures interactively.
Each notebook is self-contained and imports from the installed `fkp` package.

## Setup

```bash
pip install -e ".[viz]"   # installs matplotlib, seaborn, jupyter
jupyter lab               # or: jupyter notebook
```

---

## Notebooks

| Notebook | Figure | Description |
|---|---|---|
| `pcls_scatter.ipynb` | Figure 2 | PCLS vs teacher-edge agreement scatter plot |

---

## `pcls_scatter.ipynb` — Figure 2

Demonstrates that PCLS ≥ 0.8 reliably predicts ≥ 95% teacher-edge agreement rate.

**What it does:**
1. Simulates (teacher, dataset) pairs with varying levels of non-linearity.
2. Computes PCLS for each pair using the real `compute_pcls()` implementation.
3. Runs FKP compression at rank p=32 and measures empirical agreement.
4. Plots the scatter and overlays the PCLS threshold line.
5. Prints the Pearson correlation between PCLS and agreement.

**To run with real experiment data:**
Replace the `simulate_pair()` calls with actual runs from `outputs/*/metadata.json`.

---

## Adding new notebooks

Follow these conventions:
- Import `seed_everything(42)` as the first cell.
- Use only the public `fkp.*` API (no internal imports).
- Save figures to `../figures/` (relative to notebook directory).
- Keep notebooks clean: restart kernel and run all cells before committing.
