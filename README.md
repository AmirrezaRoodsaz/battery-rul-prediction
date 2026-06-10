# Battery Remaining-Useful-Life (RUL) Prediction

**Predict a lithium-ion cell's total cycle life from only its first ~100 charge–discharge cycles — before meaningful capacity fade is visible.**

> _Headline result pending Phase 3 (model evaluation). Target: match or beat the published ~9 % test error from Severson et al., Nature Energy 2019._

<!-- Phase 3 will insert the parity plot (predicted vs actual cycle life) here as visual proof. -->

---

## Problem & motivation

Validating a battery cell's lifetime by cycling it to failure takes months to years. If we can predict cycle life from just the first few cycles, we unlock far faster cell screening, manufacturing QA, second-life sorting, and warranty modeling. This is both a real industry problem and a recognized ML benchmark.

**Primary task:** regression — predict total cycle life (cycles to End-of-Life, defined as capacity dropping to 80 % of nominal) from features extracted from the first ~100 cycles.

## Dataset

**Severson / MIT–Stanford / Toyota Research (2019)** — 124 commercial LFP/graphite cells (A123 APR18650M1A, 1.1 Ah), fast-charged under 72 policies at 30 °C, cycled to failure. Observed cycle lives span ~150–2300 cycles. Public via the Toyota Research Institute release (`data.matr.io`). See [`data/README.md`](data/README.md) for exact provenance, license, and how to obtain it. Raw data is **not** committed.

> Severson, K.A., Attia, P.M., et al. *"Data-driven prediction of battery cycle life before capacity degradation."* **Nature Energy 4, 383–391 (2019).**

## Approach

A reproducible pipeline, not a single notebook:

1. **Acquire** raw `.mat` batches → `data/raw/` (gitignored). — [`src/data/download.py`](src/data/download.py)
2. **Parse** to tidy per-cell/per-cycle parquet → `data/processed/`. — [`src/data/load.py`](src/data/load.py), [`preprocess.py`](src/data/preprocess.py)
3. **EDA** — capacity-fade curves, cycle-life distribution, ΔQ(V) curves. — [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)
4. **Feature sets** (modular, named): variance · discharge · full. — [`src/features/`](src/features/)
5. **Model** — regularized linear baseline (reproduces the paper) → tree ensembles → optional MLP. — [`src/models/`](src/models/)
6. **Evaluate** — RMSE/MAE/MAPE on the held-out test split, parity/residual/importance plots, honest benchmark comparison. — [`reports/results.md`](reports/results.md)

## Results

_Pending Phase 3._ Will report variance / discharge / full models × RMSE / MAE / MAPE against the 9 % benchmark.

## Reproduce

```bash
make install     # create venv + install pinned deps (Python 3.11, CPU-only)
make download     # fetch raw Severson batches (see data/README.md for license)
make all          # process -> features -> train -> evaluate
make test         # run the test suite
```

## What I learned / limitations / next steps

_Written up in Phase 4 — see [`reports/results.md`](reports/results.md) and [`reports/interview_notes.md`](reports/interview_notes.md)._

## Citation & acknowledgements

Dataset and benchmark: Severson et al., Nature Energy 2019 (full citation above). This repository is an independent reimplementation for portfolio/educational purposes; it does not redistribute the dataset.

## License

[MIT](LICENSE) © 2026 Amirreza Roodsaz
