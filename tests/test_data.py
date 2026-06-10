"""Tests for the data layer.

Two kinds:
- **Pure-logic tests** (always run): the train/test split is the highest-leakage-risk piece,
  so we test it directly on synthetic ids — no 8 GB download required.
- **Integration tests** (skipped if processed data is absent): sanity-check the real parquet
  output when it exists, so CI without data still passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.data.preprocess import _assign_split

# Real cleaned counts from the Severson recipe.
N_B1, N_B2, N_B3 = 41, 43, 40
N_TOTAL = N_B1 + N_B2 + N_B3


@pytest.fixture
def fake_ids() -> list[str]:
    return [f"c{i}" for i in range(N_TOTAL)]


def test_split_partitions_all_cells_with_no_overlap(fake_ids):
    split = _assign_split(fake_ids, N_B1, N_B2, N_B3)
    # Every cell assigned exactly once; the three splits are disjoint and cover everything.
    assert set(split) == set(fake_ids)
    buckets = {"train": set(), "test": set(), "secondary_test": set()}
    for cid, s in split.items():
        buckets[s].add(cid)
    assert buckets["train"].isdisjoint(buckets["test"])
    assert buckets["train"].isdisjoint(buckets["secondary_test"])
    assert buckets["test"].isdisjoint(buckets["secondary_test"])
    assert sum(len(v) for v in buckets.values()) == N_TOTAL


def test_split_counts_match_paper(fake_ids):
    split = _assign_split(fake_ids, N_B1, N_B2, N_B3)
    counts = pd.Series(split).value_counts()
    # The paper's split: 41 train, 43 primary test, 40 secondary test.
    assert counts["train"] == 41
    assert counts["test"] == 43
    assert counts["secondary_test"] == 40


def test_secondary_test_is_exactly_batch3(fake_ids):
    # The secondary test set must be the held-out later batch (the last N_B3 cells).
    split = _assign_split(fake_ids, N_B1, N_B2, N_B3)
    secondary = {cid for cid, s in split.items() if s == "secondary_test"}
    assert secondary == set(fake_ids[-N_B3:])


# --- Integration tests against the real processed data (skipped if not built) ---

_cells_path = PROCESSED_DIR / "cells.parquet"
requires_data = pytest.mark.skipif(
    not _cells_path.exists(), reason="processed data not built (run `make process`)"
)


@requires_data
def test_real_split_has_no_cell_in_two_splits():
    cells = pd.read_parquet(_cells_path)
    assert cells["cell_id"].is_unique
    assert cells["split"].isin(["train", "test", "secondary_test"]).all()
    assert len(cells) == N_TOTAL


@requires_data
def test_cycle_life_in_physical_range():
    cells = pd.read_parquet(_cells_path)
    # Observed Severson cycle lives span ~150-2300; guard against parse/merge regressions.
    assert cells["cycle_life"].min() > 100
    assert cells["cycle_life"].max() < 2600
    assert (cells["cycle_life"] > 0).all()


@requires_data
def test_qdlin_shapes_consistent():
    qd = np.load(PROCESSED_DIR / "qdlin.npz")
    cell_keys = [k for k in qd.files if not k.startswith("_")]
    sample = qd[cell_keys[0]]
    assert sample.shape[1] == 1000  # fixed voltage grid
    assert sample.shape[0] <= 100  # only early cycles retained
    assert qd["_vdlin"].shape == (1000,)  # shared voltage axis stored once
