"""Train models on each feature set and record held-out metrics.

The validation protocol — the part interviewers probe hardest — is:

1. Fit and select hyperparameters using **5-fold cross-validation on the training split only**.
2. Refit the best pipeline on all training cells.
3. Report metrics **once** on the untouched **primary test** and **secondary test** splits.

The target is **log(cycle life)** (heavy-tailed → log-space regression); predictions are
exponentiated back to cycles before scoring, so all reported errors are in real cycle counts.
Models, the best hyperparameters, metrics, and per-cell predictions are persisted so the
evaluation/report stage can render figures without retraining.
"""

from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

from src.config import MODELS_DIR, RANDOM_SEED, REPORTS_DIR
from src.features.build_features import FEATURE_SETS, build_feature_table
from src.models.metrics import predictions_to_cycles, regression_metrics
from src.models.models import get_model

# Models trained by `make train`, in escalation order: the honest linear floor first, then
# tree ensembles. Lasso is omitted (ElasticNet generalizes it); XGBoost requires libomp.
DEFAULT_MODELS = ["elasticnet", "random_forest", "gradient_boosting", "xgboost"]
CV_FOLDS = 5
SPLITS = ("train", "test", "secondary_test")


def _xy(table: pd.DataFrame, split: str, feature_cols: list[str]):
    sub = table[table["split"] == split]
    return sub[feature_cols].to_numpy(float), sub["log_cycle_life"].to_numpy(float), sub.index


def train_one(feature_set: str, model_name: str, table: pd.DataFrame | None = None):
    """Train one (feature_set, model) combo. Returns (fitted_estimator, metrics_rows, pred_rows)."""
    table = build_feature_table(feature_set) if table is None else table
    feature_cols = FEATURE_SETS[feature_set]

    x_tr, y_tr, _ = _xy(table, "train", feature_cols)
    pipe, grid = get_model(model_name)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        search = GridSearchCV(pipe, grid, cv=CV_FOLDS, scoring="neg_mean_squared_error", n_jobs=-1)
        search.fit(x_tr, y_tr)
    best = search.best_estimator_

    metrics_rows, pred_rows = [], []
    for split in SPLITS:
        x, y_log, ids = _xy(table, split, feature_cols)
        y_true = predictions_to_cycles(y_log)
        y_pred = predictions_to_cycles(best.predict(x))
        m = regression_metrics(y_true, y_pred)
        metrics_rows.append(
            {
                "feature_set": feature_set,
                "model": model_name,
                "split": split,
                **m.as_dict(),
                "n": len(ids),
                "best_params": str(search.best_params_),
            }
        )
        pred_rows.extend(
            {
                "cell_id": cid,
                "feature_set": feature_set,
                "model": model_name,
                "split": split,
                "y_true": float(yt),
                "y_pred": float(yp),
            }
            for cid, yt, yp in zip(ids, y_true, y_pred, strict=False)
        )
    return best, metrics_rows, pred_rows


def train_all(models: list[str] | None = None) -> pd.DataFrame:
    """Train every (feature_set x model) combo; persist artifacts, metrics, and predictions."""
    models = models or DEFAULT_MODELS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    all_metrics, all_preds = [], []
    for feature_set in FEATURE_SETS:
        table = build_feature_table(feature_set)
        for model_name in models:
            best, metrics_rows, pred_rows = train_one(feature_set, model_name, table)
            joblib.dump(best, MODELS_DIR / f"{feature_set}__{model_name}.joblib")
            all_metrics.extend(metrics_rows)
            all_preds.extend(pred_rows)
            test_mape = next(r["mape"] for r in metrics_rows if r["split"] == "test")
            print(f"{feature_set:>9} x {model_name:<17} primary-test MAPE = {test_mape:5.1f}%")

    metrics_df = pd.DataFrame(all_metrics)
    preds_df = pd.DataFrame(all_preds)
    metrics_df.to_parquet(REPORTS_DIR / "metrics.parquet", index=False)
    preds_df.to_parquet(REPORTS_DIR / "predictions.parquet", index=False)
    print(f"\nSaved metrics ({len(metrics_df)} rows) and predictions to {REPORTS_DIR}/.")
    return metrics_df


if __name__ == "__main__":
    train_all()
