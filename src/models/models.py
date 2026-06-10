"""Model registry: each entry is a scikit-learn pipeline + a hyperparameter grid.

We start with the **regularized linear baseline** (the honest floor that reproduces the paper)
and escalate to tree ensembles. Every model is wrapped in a ``StandardScaler`` pipeline so that:
- feature scaling happens *inside* each cross-validation fold (no leakage), and
- linear coefficients are comparable across features.

All randomness is seeded from ``src.config.RANDOM_SEED`` for reproducibility.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

# Friendly display names for reports.
DISPLAY_NAMES = {
    "elasticnet": "ElasticNet (linear)",
    "lasso": "Lasso (linear)",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def _pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def get_model(name: str) -> tuple[Pipeline, dict]:
    """Return ``(pipeline, param_grid)`` for ``name``. The grid is searched with CV on train only."""
    if name == "elasticnet":
        pipe = _pipe(ElasticNet(max_iter=100_000, random_state=RANDOM_SEED))
        grid = {
            "model__alpha": np.logspace(-4, 1, 30),
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
        }
        return pipe, grid

    if name == "lasso":
        pipe = _pipe(Lasso(max_iter=100_000, random_state=RANDOM_SEED))
        grid = {"model__alpha": np.logspace(-4, 1, 50)}
        return pipe, grid

    if name == "random_forest":
        pipe = _pipe(RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1))
        grid = {
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 3, 5],
            "model__min_samples_leaf": [1, 2, 4],
        }
        return pipe, grid

    if name == "gradient_boosting":
        pipe = _pipe(GradientBoostingRegressor(random_state=RANDOM_SEED))
        grid = {
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3],
        }
        return pipe, grid

    if name == "xgboost":
        # Imported lazily so the linear baselines run even if libomp/xgboost is unavailable.
        from xgboost import XGBRegressor

        pipe = _pipe(
            XGBRegressor(
                random_state=RANDOM_SEED,
                n_jobs=-1,
                objective="reg:squarederror",
                tree_method="hist",
            )
        )
        grid = {
            "model__n_estimators": [300, 600],
            "model__learning_rate": [0.02, 0.05, 0.1],
            "model__max_depth": [2, 3],
            "model__subsample": [0.8, 1.0],
        }
        return pipe, grid

    raise KeyError(f"unknown model {name!r}; choose from {list(DISPLAY_NAMES)}")
