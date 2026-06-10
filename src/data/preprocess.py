"""Assemble raw batches into a clean, tidy, regenerable processed dataset.

This reproduces the exact data-cleaning recipe from the paper's official repository, which is
essential for a faithful benchmark comparison:

1. **Exclude cells that never reach 80 % capacity** (batch 1: 5 cells) — they have no defined
   End-of-Life, so cycle life is undefined.
2. **Merge cross-batch continuations.** Five cells were physically cycled across batch 1 *and*
   batch 2. Their true cycle life is the sum, so we add the continuation length back to the
   batch-1 cell's cycle life and concatenate their cycle records, then drop the batch-2 copies.
3. **Drop noisy channels** (batch 3: 6 cells).
4. **Apply the canonical train / primary-test / secondary-test split** by position, exactly as
   the authors defined it. Respecting this split is what makes our reported numbers comparable
   to the published ~9 % benchmark — and avoids the leakage interviewers probe for.

Outputs (all gitignored, regenerable via ``make process``):
- ``cells.parquet``   — one row per cell: id, batch, cycle_life (target), policy, split.
- ``summary.parquet`` — long format, one row per (cell, cycle): QD/QC/IR/temps/chargetime.
- ``qdlin.npz``       — per-cell (n_cycles, 1000) discharge Q(V) curves for the first 100 cycles
                        (stored as a tensor archive; it is not naturally tabular).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, RAW_DIR, ensure_dirs
from src.data.load import Cell, parse_batch, read_vdlin

RAW_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

# Cells that never reach 80 % capacity (no defined EOL) — removed.
EXCLUDE_BATCH1 = ("b1c8", "b1c10", "b1c12", "b1c13", "b1c22")
# Noisy channels in batch 3 — removed.
EXCLUDE_BATCH3 = ("b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43")
# Cross-batch continuations: (batch1_cell, batch2_cell, extra_cycles_to_add).
CONTINUATIONS = (
    ("b1c0", "b2c7", 662),
    ("b1c1", "b2c8", 981),
    ("b1c2", "b2c9", 1060),
    ("b1c3", "b2c15", 208),
    ("b1c4", "b2c16", 482),
)


def _merge_continuations(b1: dict[str, Cell], b2: dict[str, Cell]) -> None:
    """Fold batch-2 continuation cells back into their batch-1 originals (in place)."""
    for b1_key, b2_key, add_len in CONTINUATIONS:
        cell, cont = b1[b1_key], b2[b2_key]
        cell.cycle_life += add_len
        for field, arr in cell.summary.items():
            if field == "cycle":
                # Continuation cycle numbers continue from where batch 1 left off.
                cell.summary[field] = np.hstack([arr, cont.summary[field] + len(arr)])
            else:
                cell.summary[field] = np.hstack([arr, cont.summary[field]])
        del b2[b2_key]


def _assign_split(cell_ids: list[str], n_b1: int, n_b2: int, n_b3: int) -> dict[str, str]:
    """Reproduce the authors' index-based split over the ordered (b1, b2, b3) cell list."""
    n_primary = n_b1 + n_b2  # batches 1+2 form train + primary test
    n_total = n_primary + n_b3
    train_idx = set(range(1, n_primary - 1, 2))
    primary_idx = set(range(0, n_primary, 2)) | {n_primary - 1}
    secondary_idx = set(range(n_total - n_b3, n_total))
    split = {}
    for i, cid in enumerate(cell_ids):
        if i in train_idx:
            split[cid] = "train"
        elif i in primary_idx:
            split[cid] = "test"  # primary test
        elif i in secondary_idx:
            split[cid] = "secondary_test"
        else:  # pragma: no cover - all indices covered by construction
            raise ValueError(f"cell index {i} ({cid}) fell outside every split")
    return split


def build_processed() -> None:
    ensure_dirs()
    print("Parsing raw batches (this reads ~8 GB of HDF5; ~1 min)...")
    b1 = parse_batch(RAW_DIR / RAW_FILES[1], batch=1)
    b2 = parse_batch(RAW_DIR / RAW_FILES[2], batch=2)
    b3 = parse_batch(RAW_DIR / RAW_FILES[3], batch=3)

    # 1. Exclude undefined-EOL cells from batch 1.
    for cid in EXCLUDE_BATCH1:
        del b1[cid]
    # 2. Merge cross-batch continuations (also removes them from batch 2).
    _merge_continuations(b1, b2)
    # 3. Drop noisy batch-3 channels.
    for cid in EXCLUDE_BATCH3:
        del b3[cid]

    n_b1, n_b2, n_b3 = len(b1), len(b2), len(b3)
    print(
        f"After cleaning: batch1={n_b1}, batch2={n_b2}, batch3={n_b3}, total={n_b1 + n_b2 + n_b3}"
    )

    merged: dict[str, Cell] = {**b1, **b2, **b3}
    cell_ids = list(merged.keys())
    split = _assign_split(cell_ids, n_b1, n_b2, n_b3)

    # --- cells.parquet (one row per cell) ---
    cells_df = pd.DataFrame(
        {
            "cell_id": cid,
            "batch": c.batch,
            "cycle_life": c.cycle_life,
            "charge_policy": c.charge_policy,
            "split": split[cid],
        }
        for cid, c in merged.items()
    )

    # --- summary.parquet (long: one row per cell-cycle) ---
    summary_frames = []
    for cid, c in merged.items():
        df = pd.DataFrame(c.summary)
        df.insert(0, "cell_id", cid)
        summary_frames.append(df)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    summary_df["cycle"] = summary_df["cycle"].astype(int)

    # --- qdlin.npz (per-cell tensor of early-cycle Q(V) curves + shared voltage grid) ---
    qdlin_arrays = {cid: c.qdlin.astype(np.float32) for cid, c in merged.items()}
    # The Vdlin voltage grid is shared across all cells; store once under a reserved key.
    qdlin_arrays["_vdlin"] = read_vdlin(RAW_DIR / RAW_FILES[1]).astype(np.float32)

    cells_path = PROCESSED_DIR / "cells.parquet"
    summary_path = PROCESSED_DIR / "summary.parquet"
    qdlin_path = PROCESSED_DIR / "qdlin.npz"
    cells_df.to_parquet(cells_path, index=False)
    summary_df.to_parquet(summary_path, index=False)
    np.savez_compressed(qdlin_path, **qdlin_arrays)

    print(
        f"Wrote {cells_path.name} ({len(cells_df)} cells), "
        f"{summary_path.name} ({len(summary_df):,} cell-cycles), {qdlin_path.name}."
    )
    print("Split counts:\n" + cells_df["split"].value_counts().to_string())


if __name__ == "__main__":
    build_processed()
