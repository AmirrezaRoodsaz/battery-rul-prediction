"""Regression metrics, computed in **cycle-life space** (not log space).

We train on log cycle life (the target is heavy-tailed), but report errors on the real cycle
counts so they are interpretable and comparable to the paper's ~9 % benchmark. MAPE is the headline
number the paper reports; RMSE/MAE give absolute context in cycles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Metrics:
    rmse: float  # root mean squared error, cycles
    mae: float  # mean absolute error, cycles
    mape: float  # mean absolute percentage error, %

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def regression_metrics(y_true_cycles: np.ndarray, y_pred_cycles: np.ndarray) -> Metrics:
    """Compute RMSE / MAE / MAPE on cycle counts.

    Inputs are actual cycle lives (already exponentiated back from log space if needed).
    """
    y_true = np.asarray(y_true_cycles, dtype=float)
    y_pred = np.asarray(y_pred_cycles, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / y_true) * 100.0)
    return Metrics(rmse=rmse, mae=mae, mape=mape)


def predictions_to_cycles(log_pred: np.ndarray) -> np.ndarray:
    """Invert the natural-log target transform back to cycle counts."""
    return np.exp(np.asarray(log_pred, dtype=float))
