"""Tests for feature engineering.

Pure-math tests (always run) pin down the ΔQ(V) statistics on synthetic inputs. Integration
tests (skipped without processed data) check the assembled feature tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CYCLE_EARLY, CYCLE_LATE, PROCESSED_DIR
from src.features import build_features as bf


def test_delta_q_takes_cycle_100_minus_cycle_10_and_drops_nan():
    qdlin = np.zeros((100, 5))
    qdlin[CYCLE_LATE - 1] = [1.0, 2.0, 3.0, 4.0, np.nan]  # row 99
    qdlin[CYCLE_EARLY - 1] = [0.0, 0.0, 0.0, 0.0, 0.0]  # row 9
    dq = bf.delta_q(qdlin)
    # NaN column dropped; remaining values are late - early.
    assert dq.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_dq_variance_is_log10_of_variance():
    dq = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.log10(dq.var() + 1e-10)
    assert bf.dq_variance(dq) == pytest.approx(expected)


def test_dq_minimum_uses_absolute_value():
    dq = np.array([-0.5, 0.1, 0.2])
    assert bf.dq_minimum(dq) == pytest.approx(np.log10(0.5 + 1e-10))


def test_capacity_fade_slope_recovers_known_line():
    # QD = 1.1 - 0.001 * cycle over cycles 2..100 -> slope should be -0.001.
    cycles = np.arange(2, 101)
    summary = pd.DataFrame({"cycle": cycles, "QD": 1.1 - 0.001 * cycles})
    assert bf.capacity_fade_slope(summary) == pytest.approx(-0.001, abs=1e-6)


def test_feature_sets_have_expected_sizes():
    assert len(bf.FEATURE_SETS["variance"]) == 1
    assert len(bf.FEATURE_SETS["discharge"]) == 6
    assert len(bf.FEATURE_SETS["full"]) == 9


# --- Integration tests against real processed data ---

requires_data = pytest.mark.skipif(
    not (PROCESSED_DIR / "cells.parquet").exists(),
    reason="processed data not built (run `make process`)",
)


@requires_data
@pytest.mark.parametrize("feature_set", ["variance", "discharge", "full"])
def test_feature_table_shape_and_finiteness(feature_set):
    table = bf.build_feature_table(feature_set)
    feat_cols = bf.FEATURE_SETS[feature_set]
    assert len(table) == 124
    assert list(table.columns) == feat_cols + ["cycle_life", "log_cycle_life", "split"]
    # No NaN/inf leaks into the model inputs.
    assert np.isfinite(table[feat_cols].to_numpy()).all()


@requires_data
def test_log_target_matches_log_of_cycle_life():
    table = bf.build_feature_table("variance")
    assert np.allclose(table["log_cycle_life"], np.log(table["cycle_life"]))
