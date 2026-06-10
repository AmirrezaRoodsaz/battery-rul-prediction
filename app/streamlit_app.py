"""Minimal Streamlit demo for battery cycle-life prediction.

Two tabs:
- **Dataset cell** — pick a held-out Severson cell and see predicted vs actual cycle life.
- **Upload your own CSV** — upload cycle-10 and cycle-100 discharge Q(V) curves and get a prediction
  from the variance model (the single-feature model that needs *only* those two curves).

Deliberately small and secondary to the analysis. Run with ``make app`` (after ``make pipeline``).
"""

from __future__ import annotations

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import MODELS_DIR
from src.data.dataset import load_processed
from src.features import build_features as bf

st.set_page_config(page_title="Battery RUL Prediction", layout="centered")
st.title("🔋 Battery cycle-life prediction")
st.caption(
    "Predicts a cell's total cycle life from its first 100 cycles "
    "(Severson 2019 dataset). Demo of the models in this repo — see the README for the analysis."
)

# Headline model (for the dataset tab) and the variance model (for the upload tab).
HEADLINE_FS, HEADLINE_MODEL = "discharge", "gradient_boosting"
HEADLINE_PATH = MODELS_DIR / f"{HEADLINE_FS}__{HEADLINE_MODEL}.joblib"
VARIANCE_PATH = MODELS_DIR / "variance__elasticnet.joblib"

# Voltage grid the dataset's Qdlin curves use; uploads are interpolated onto it.
VDLIN_MIN, VDLIN_MAX, VDLIN_N = 2.0, 3.5, 1000


@st.cache_resource
def _load():
    ds = load_processed()
    table = bf.build_feature_table(HEADLINE_FS, ds)
    headline_model = joblib.load(HEADLINE_PATH)
    variance_model = joblib.load(VARIANCE_PATH)
    return ds, table, headline_model, variance_model


if not HEADLINE_PATH.exists() or not VARIANCE_PATH.exists():
    st.error("Trained models not found. Run `make pipeline` first.")
    st.stop()

ds, table, headline_model, variance_model = _load()

tab_dataset, tab_upload = st.tabs(["📂 Dataset cell", "⬆️ Upload your own CSV"])

# ----------------------------------------------------------------------------------------
# Tab 1 — pick a held-out dataset cell
# ----------------------------------------------------------------------------------------
with tab_dataset:
    test_ids = table.index[table["split"].isin(["test", "secondary_test"])].tolist()
    cell_id = st.selectbox("Choose a held-out cell", sorted(test_ids))

    row = table.loc[cell_id]
    policy = ds.cells.set_index("cell_id").loc[cell_id, "charge_policy"]

    x = row[bf.FEATURE_SETS[HEADLINE_FS]].to_numpy(float).reshape(1, -1)
    pred_cycles = float(np.exp(headline_model.predict(x)[0]))
    actual = float(row["cycle_life"])
    err_pct = abs(pred_cycles - actual) / actual * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted cycle life", f"{pred_cycles:,.0f}")
    c2.metric("Actual cycle life", f"{actual:,.0f}")
    c3.metric("Error", f"{err_pct:.1f} %")
    st.caption(f"Model: {HEADLINE_FS} features × Gradient Boosting · charge policy: {policy}")

    q = ds.qdlin[cell_id]
    dq = q[99] - q[9]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ds.vdlin, dq, color="#4C72B0")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"$\Delta Q_{100-10}(V)$ (Ah)")
    ax.set_title(f"{cell_id}: ΔQ(V) between cycle 10 and 100")
    fig.tight_layout()
    st.pyplot(fig)


# ----------------------------------------------------------------------------------------
# Tab 2 — upload your own Q(V) curves
# ----------------------------------------------------------------------------------------
REQUIRED_COLUMNS = ["voltage", "q_cycle10", "q_cycle100"]


def _example_csv() -> str:
    """Build an example CSV from a real dataset cell so users can see the exact format."""
    example_id = sorted(ds.qdlin)[0]
    q = ds.qdlin[example_id]
    df = pd.DataFrame({"voltage": ds.vdlin, "q_cycle10": q[9], "q_cycle100": q[99]}).dropna()
    return df.to_csv(index=False)


def _predict_from_curves(df: pd.DataFrame) -> dict:
    """Interpolate uploaded curves onto the model's voltage grid, compute ΔQ features, predict."""
    grid = np.linspace(VDLIN_MIN, VDLIN_MAX, VDLIN_N)
    v = df["voltage"].to_numpy(float)
    order = np.argsort(v)  # np.interp needs ascending x
    q10 = np.interp(grid, v[order], df["q_cycle10"].to_numpy(float)[order])
    q100 = np.interp(grid, v[order], df["q_cycle100"].to_numpy(float)[order])
    dq = q100 - q10

    feats = {
        "dq_variance": bf.dq_variance(dq),
        "dq_minimum": bf.dq_minimum(dq),
        "dq_skewness": bf.dq_skewness(dq),
        "dq_kurtosis": bf.dq_kurtosis(dq),
    }
    x = np.array([[feats["dq_variance"]]])  # variance model = single feature
    pred = float(np.exp(variance_model.predict(x)[0]))
    return {"pred": pred, "features": feats, "grid": grid, "dq": dq}


with tab_upload:
    st.markdown(
        "Upload your **own** cycle-10 and cycle-100 discharge curves and the **variance model** "
        "predicts cycle life from them."
    )
    st.download_button(
        "⬇️ Download an example CSV (correct format)",
        data=_example_csv(),
        file_name="example_qv_curves.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload a CSV", type="csv")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            st.error(f"CSV is missing required column(s): {missing}. Expected: {REQUIRED_COLUMNS}.")
            st.stop()
        df = df[REQUIRED_COLUMNS].dropna()
        if len(df) < 10:
            st.error("Need at least ~10 valid voltage points to form a curve.")
            st.stop()

        result = _predict_from_curves(df)
        st.metric("Predicted cycle life", f"{result['pred']:,.0f} cycles")
        st.caption("Model: variance feature × ElasticNet (the single-feature baseline).")

        feats = result["features"]
        st.write(
            f"Computed ΔQ(V) features → "
            f"log₁₀ var = **{feats['dq_variance']:.2f}**, "
            f"log₁₀|min| = {feats['dq_minimum']:.2f}, "
            f"skew = {feats['dq_skewness']:.2f}, kurt = {feats['dq_kurtosis']:.2f}"
        )

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(result["grid"], result["dq"], color="#DD8452")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel(r"$\Delta Q_{100-10}(V)$ (Ah)")
        ax.set_title("Your uploaded ΔQ(V) curve")
        fig.tight_layout()
        st.pyplot(fig)

    # ---- Explanation: what the CSV is, the exact format, units, and limits ----
    st.divider()
    st.subheader("📄 What the CSV must contain")
    st.markdown(
        """
**The file is a single CSV with exactly three columns** describing two discharge curves of the
*same cell*, measured at cycle 10 and cycle 100:

| Column | Meaning | Unit |
|---|---|---|
| `voltage` | The voltage points the capacity was measured at | Volts (V) |
| `q_cycle10` | Discharge capacity at that voltage, **at cycle 10** | Amp-hours (Ah) |
| `q_cycle100` | Discharge capacity at that voltage, **at cycle 100** | Amp-hours (Ah) |

**One row per voltage point.** Example (first rows):

```csv
voltage,q_cycle10,q_cycle100
2.00,1.0731,1.0640
2.0015,1.0729,1.0637
2.0030,1.0727,1.0635
...
3.50,0.0000,0.0000
```

**Format rules**
- Voltages should cover roughly **2.0–3.5 V** (the model's discharge window). Any monotonic grid is
  fine — the app interpolates your curve onto its internal 1000-point grid. Points outside the range
  are clamped.
- The number of rows is flexible (≈100–1000 is plenty); they don't have to match the internal grid.
- Capacity is the **discharge** capacity Q at each voltage, in Ah, for a single cell.
- Use the **Download example CSV** button above to get a correctly-formatted file you can edit.

**What the app does with it:** it computes **ΔQ(V) = q_cycle100 − q_cycle10**, takes
`log₁₀(variance)` of that curve, and feeds that single number to the variance model. That curve's
*shape change* between cycle 10 and 100 is the degradation fingerprint that predicts lifetime.

**Why only the variance model here?** The stronger `discharge`/`full` models also need per-cycle
*discharge-capacity-vs-cycle* data (capacity at cycle 2, fade slope, charge time, internal
resistance) which a two-curve CSV doesn't contain. The variance model needs **only** these two
curves — so it's the honest fit for this upload format.

**Important — what this is NOT:** the curves must come from **controlled lab cycling of a single cell**
(like the Severson A123 LFP cells). Data from an EV's OBD-II/CAN bus (pack-level voltage/current
during real driving, different chemistry, uncontrolled temperature, no clean per-cycle Q(V)) is a
**different kind of measurement** and will not give a meaningful prediction here — that is a separate
State-of-Health problem, not this cell-screening model.
"""
    )
