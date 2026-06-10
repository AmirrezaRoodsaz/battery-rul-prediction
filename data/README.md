# Data provenance

This directory holds the dataset locally. **Nothing here is committed to git** except this
README and `.gitkeep` markers — `data/raw/` and `data/processed/` are gitignored because the
data is large, regenerable, and license-bound.

## Source

**Severson / MIT–Stanford / Toyota Research Institute battery cycle-life dataset (2019).**

- Paper: Severson, K.A., Attia, P.M., Jin, N., et al. *"Data-driven prediction of battery
  cycle life before capacity degradation."* **Nature Energy 4, 383–391 (2019).**
  https://doi.org/10.1038/s41560-019-0356-8
- Data release: Toyota Research Institute, hosted at `https://data.matr.io/1/`.
- Contents: 124 commercial LFP/graphite cells (A123 APR18650M1A, 1.1 Ah nominal), fast-charged
  under 72 charging policies at 30 °C, cycled to failure. Delivered as **3 batches** in MATLAB
  v7.3 `.mat` files (which are HDF5 under the hood, readable with `h5py`):
  - `2017-05-12` (batch 1)
  - `2017-06-30` (batch 2)
  - `2018-04-12` (batch 3)

## License

The dataset is released by Toyota Research Institute for research use. **Verify the current
license terms on the data portal before redistributing.** This repository does **not** include
or redistribute the data — it only provides code to download and process it. Cite the paper
above in any derived work.

## How to obtain

Preferred (scripted):

```bash
make download        # runs src/data/download.py -> populates data/raw/
```

If the portal layout changes or scripted fetch is blocked, the download script prints the exact
manual-download URLs and target filenames. Place the downloaded `.mat` files in `data/raw/`.

## Processed form

`make process` parses the raw `.mat` batches into a tidy, regenerable parquet form in
`data/processed/` (one row per cell-cycle plus a per-cell summary table). This is also
gitignored — regenerate it from raw at any time.

## Known data notes (from the paper / community)

- A small number of cells are flagged as anomalous and excluded by the original authors;
  the loader documents which and why.
- The dataset ships with a defined **train / primary-test / secondary-test** split across the
  batches. We respect it strictly — no leakage across the split. See `src/data/load.py`.
