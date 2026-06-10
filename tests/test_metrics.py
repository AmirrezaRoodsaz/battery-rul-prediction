"""Tests for the regression metrics — small, exact checks on known inputs."""

from __future__ import annotations

import numpy as np

from src.models.metrics import predictions_to_cycles, regression_metrics


def test_perfect_predictions_give_zero_error():
    y = np.array([100.0, 500.0, 1500.0])
    m = regression_metrics(y, y)
    assert m.rmse == 0.0
    assert m.mae == 0.0
    assert m.mape == 0.0


def test_mape_is_percentage_of_true():
    # A constant 10% over-prediction should give MAPE = 10, regardless of magnitude.
    y_true = np.array([100.0, 1000.0])
    y_pred = y_true * 1.1
    m = regression_metrics(y_true, y_pred)
    assert m.mape == np.float64(10.0) or abs(m.mape - 10.0) < 1e-9


def test_rmse_penalizes_large_errors_more_than_mae():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([100.0, 200.0])  # one big miss
    m = regression_metrics(y_true, y_pred)
    assert m.rmse > m.mae


def test_predictions_to_cycles_inverts_natural_log():
    cycles = np.array([150.0, 800.0, 2300.0])
    assert np.allclose(predictions_to_cycles(np.log(cycles)), cycles)
