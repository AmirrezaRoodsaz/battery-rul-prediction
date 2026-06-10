"""Convenience accessors for the processed dataset.

Every downstream stage (EDA, features, modeling) loads the data through here, so they all
agree on the schema and the reserved ``_vdlin`` key handling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR

VDLIN_KEY = "_vdlin"


@dataclass
class Dataset:
    cells: pd.DataFrame  # one row per cell: cell_id, batch, cycle_life, charge_policy, split
    summary: pd.DataFrame  # long: cell_id, cycle, QD, QC, IR, Tavg, Tmin, Tmax, chargetime
    qdlin: dict[str, np.ndarray]  # cell_id -> (n_cycles, 1000) early-cycle Q(V) curves
    vdlin: np.ndarray  # shared (1000,) voltage grid the Qdlin curves are sampled on

    def split_ids(self, split: str) -> list[str]:
        return self.cells.loc[self.cells["split"] == split, "cell_id"].tolist()


def _require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `make download && make process` to build the processed data."
        )
    return path


def load_processed() -> Dataset:
    cells = pd.read_parquet(_require(PROCESSED_DIR / "cells.parquet"))
    summary = pd.read_parquet(_require(PROCESSED_DIR / "summary.parquet"))
    npz = np.load(_require(PROCESSED_DIR / "qdlin.npz"))
    vdlin = npz[VDLIN_KEY]
    qdlin = {k: npz[k] for k in npz.files if k != VDLIN_KEY}
    return Dataset(cells=cells, summary=summary, qdlin=qdlin, vdlin=vdlin)
