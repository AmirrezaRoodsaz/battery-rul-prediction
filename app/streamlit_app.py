"""Minimal Streamlit demo — select a cell, see its predicted vs actual cycle life.

Deliberately small and secondary to the analysis. It loads the processed data and the trained
headline model, lets you pick a held-out cell, and shows the early-cycle ΔQ(V) curve alongside the
prediction. Run with: ``make app`` (after ``make pipeline``).
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import MODELS_DIR
from src.data.dataset import load_processed
from src.features.build_features import FEATURE_SETS, build_feature_table

st.set_page_config(page_title="Battery RUL Prediction", layout="centered")
st.title("🔋 Battery cycle-life prediction")
st.caption(
    "Predicts a cell's total cycle life from its first 100 cycles "
    "(Severson 2019 dataset). Demo of the model in this repo — see the README for the analysis."
)

# Headline model from the repo: discharge features × Gradient Boosting.
FEATURE_SET, MODEL = "discharge", "gradient_boosting"
MODEL_PATH = MODELS_DIR / f"{FEATURE_SET}__{MODEL}.joblib"


@st.cache_resource
def _load():
    ds = load_processed()
    table = build_feature_table(FEATURE_SET, ds)
    model = joblib.load(MODEL_PATH)
    return ds, table, model


if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `make pipeline` first.")
    st.stop()

ds, table, model = _load()

# Let the user pick a held-out (test) cell — the honest setting.
test_ids = table.index[table["split"].isin(["test", "secondary_test"])].tolist()
cell_id = st.selectbox("Choose a held-out cell", sorted(test_ids))

row = table.loc[cell_id]
policy = ds.cells.set_index("cell_id").loc[cell_id, "charge_policy"]

x = row[FEATURE_SETS[FEATURE_SET]].to_numpy(float).reshape(1, -1)
pred_cycles = float(np.exp(model.predict(x)[0]))
actual = float(row["cycle_life"])
err_pct = abs(pred_cycles - actual) / actual * 100

c1, c2, c3 = st.columns(3)
c1.metric("Predicted cycle life", f"{pred_cycles:,.0f}")
c2.metric("Actual cycle life", f"{actual:,.0f}")
c3.metric("Error", f"{err_pct:.1f} %")

st.info(
    "This is a point estimate from a model that averages ~9–12 % error on held-out cells. "
    "Treat it as a screening estimate, not a guarantee — a production system should report a "
    "calibrated uncertainty interval (see the README's next-steps)."
)

# Show the early-cycle ΔQ(V) curve that drives the prediction.
st.subheader("Early-life signal: ΔQ(V) between cycle 10 and 100")
q = ds.qdlin[cell_id]
dq = q[99] - q[9]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(ds.vdlin, dq, color="#4C72B0")
ax.set_xlabel("Voltage (V)")
ax.set_ylabel(r"$\Delta Q_{100-10}(V)$ (Ah)")
ax.set_title(f"{cell_id} — charge policy: {policy}")
fig.tight_layout()
st.pyplot(fig)
