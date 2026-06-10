r"""Feature engineering — the three named feature sets from Severson et al. (2019).

Every model in this project is trained on features extracted from **only the first 100 cycles**,
which is what makes early cycle-life prediction useful. We build up predictive power in three
nested, increasingly rich sets so the contribution of each idea is visible:

- **variance** (1 feature):  ``log10 var(ΔQ_{100-10}(V))`` — the single canonical feature.
- **discharge** (6 features): statistics of the ΔQ(V) curve + early discharge-capacity features.
- **full** (9 features):      adds capacity-fade-curve fit, charge time, temperature, and
                              internal-resistance features.

The central object is **ΔQ(V) = Q_discharge(cycle 100) − Q_discharge(cycle 10)** on the fixed
1000-point voltage grid. Capturing *how the discharge curve changes shape* early in life encodes
degradation-mode information that predicts eventual lifetime — long before capacity fade is
visible. Feature transforms (log10 of curve statistics, etc.) mirror the paper so our numbers are
comparable to the published ~9 % benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CYCLE_EARLY, CYCLE_LATE
from src.data.dataset import Dataset, load_processed

# Cycle windows (1-based cycle numbers) used by the capacity-fade / charge-time / temperature /
# IR features, matching the paper's "first 100 cycles" framing.
_FADE_LO, _FADE_HI = 2, 100  # capacity-fade fit window
_CHARGE_CYCLES = (2, 6)  # average charge time over early cycles
_EPS = 1e-10  # keeps log() finite when an argument is ~0

# Feature-set definitions: ordered lists of feature names.
FEATURE_SETS: dict[str, list[str]] = {
    "variance": ["dq_variance"],
    "discharge": [
        "dq_minimum",
        "dq_variance",
        "dq_skewness",
        "dq_kurtosis",
        "early_discharge_capacity",
        "max_minus_early_capacity",
    ],
    "full": [
        "dq_minimum",
        "dq_variance",
        "capacity_fade_slope",
        "capacity_fade_intercept",
        "early_discharge_capacity",
        "avg_early_charge_time",
        "mean_temperature",
        "min_internal_resistance",
        "internal_resistance_change",
    ],
}


# --- ΔQ(V) curve and its statistics -------------------------------------------------------


def delta_q(qdlin: np.ndarray) -> np.ndarray:
    """ΔQ(V) between cycle 100 and cycle 10, with NaN samples dropped.

    Cycle *n* lives at row *n-1* (rows are 0-indexed by cycle). Dropping NaNs handles the rare
    degenerate cycle whose interpolated curve was stored as NaN upstream.
    """
    dq = qdlin[CYCLE_LATE - 1] - qdlin[CYCLE_EARLY - 1]
    return dq[~np.isnan(dq)]


def dq_minimum(dq: np.ndarray) -> float:
    """log10 of the magnitude of the most-negative point of ΔQ(V).

    The deepest dip marks the voltage where the cell lost the most capacity between cycles 10
    and 100 — a strong, early degradation signal."""
    return float(np.log10(np.abs(dq.min()) + _EPS))


def dq_variance(dq: np.ndarray) -> float:
    """log10 of the variance of ΔQ(V) — the single most predictive feature (paper's headline)."""
    return float(np.log10(dq.var() + _EPS))


def dq_skewness(dq: np.ndarray) -> float:
    """log10|skewness| of ΔQ(V): asymmetry of the change distribution across voltage."""
    s = ((dq - dq.mean()) ** 3).mean() / (dq.std() ** 3 + _EPS)
    return float(np.log10(np.abs(s) + _EPS))


def dq_kurtosis(dq: np.ndarray) -> float:
    """log10 kurtosis of ΔQ(V): how peaked/heavy-tailed the change is across voltage."""
    k = ((dq - dq.mean()) ** 4).mean() / (dq.var() ** 2 + _EPS)
    return float(np.log10(k + _EPS))


# --- Capacity-fade-curve features (discharge capacity vs cycle, cycles 2–100) --------------


def _qd_window(summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    win = summary[(summary["cycle"] >= _FADE_LO) & (summary["cycle"] <= _FADE_HI)]
    return win["cycle"].to_numpy(float), win["QD"].to_numpy(float)


def early_discharge_capacity(summary: pd.DataFrame) -> float:
    """Discharge capacity at cycle 2 — a cell's initial usable capacity."""
    row = summary.loc[summary["cycle"] == _FADE_LO, "QD"]
    return float(row.iloc[0]) if len(row) else 0.0


def max_minus_early_capacity(summary: pd.DataFrame) -> float:
    """Max discharge capacity (cycles 2–100) minus the cycle-2 capacity.

    A small early *rise* then fall is common; this captures the bump's size."""
    _, qd = _qd_window(summary)
    return float(qd.max() - qd[0]) if len(qd) else 0.0


def capacity_fade_slope(summary: pd.DataFrame) -> float:
    """Slope of a linear fit to discharge capacity vs cycle (cycles 2–100).

    How fast capacity is already trending down in the first 100 cycles."""
    x, qd = _qd_window(summary)
    if len(qd) < 2:
        return 0.0
    return float(np.polyfit(x, qd, 1)[0])


def capacity_fade_intercept(summary: pd.DataFrame) -> float:
    """Intercept of that same linear fit — the extrapolated cycle-0 capacity."""
    x, qd = _qd_window(summary)
    if len(qd) < 2:
        return 0.0
    return float(np.polyfit(x, qd, 1)[1])


# --- Charge-time / temperature / internal-resistance features ------------------------------


def avg_early_charge_time(summary: pd.DataFrame) -> float:
    """log of average charge time over cycles 2–6.

    Fast-charging policy and early kinetics leave a signature in how long a charge takes."""
    lo, hi = _CHARGE_CYCLES
    win = summary[(summary["cycle"] >= lo) & (summary["cycle"] <= hi)]
    ct = win["chargetime"].to_numpy(float)
    ct = ct[np.isfinite(ct)]
    return float(np.log(ct.mean() + _EPS)) if len(ct) else 0.0


def mean_temperature(summary: pd.DataFrame) -> float:
    """log of the mean per-cycle average temperature over cycles 2–100.

    A proxy for the temperature exposure the cell accumulates early in life (the paper uses an
    integral of temperature over time; we approximate it with mean cycle temperature, which is
    what is directly available in the summary statistics)."""
    _, _ = _qd_window(summary)
    win = summary[(summary["cycle"] >= _FADE_LO) & (summary["cycle"] <= _FADE_HI)]
    t = win["Tavg"].to_numpy(float)
    t = t[np.isfinite(t)]
    return float(np.log(t.mean() + _EPS)) if len(t) else 0.0


def _ir_window(summary: pd.DataFrame) -> np.ndarray:
    win = summary[(summary["cycle"] >= _FADE_LO) & (summary["cycle"] <= _FADE_HI)]
    ir = win["IR"].to_numpy(float)
    # IR is occasionally recorded as exactly 0 (placeholder); treat those as missing.
    return ir[np.isfinite(ir) & (ir > 0)]


def min_internal_resistance(summary: pd.DataFrame) -> float:
    """Minimum internal resistance over cycles 2–100. Lower IR ~ healthier early kinetics."""
    ir = _ir_window(summary)
    return float(ir.min()) if len(ir) else 0.0


def internal_resistance_change(summary: pd.DataFrame) -> float:
    """Change in internal resistance from cycle 2 to cycle 100 (late − early).

    Rising IR indicates growth of resistive interphase layers — a degradation mechanism."""
    early = summary.loc[summary["cycle"] == _FADE_LO, "IR"]
    late = summary.loc[summary["cycle"] == _FADE_HI, "IR"]
    if len(early) and len(late):
        return float(late.iloc[0] - early.iloc[0])
    return 0.0


# Map feature name -> (compute fn, whether it needs ΔQ or the summary frame).
_DQ_FEATURES = {"dq_minimum", "dq_variance", "dq_skewness", "dq_kurtosis"}
_FEATURE_FNS = {
    "dq_minimum": dq_minimum,
    "dq_variance": dq_variance,
    "dq_skewness": dq_skewness,
    "dq_kurtosis": dq_kurtosis,
    "early_discharge_capacity": early_discharge_capacity,
    "max_minus_early_capacity": max_minus_early_capacity,
    "capacity_fade_slope": capacity_fade_slope,
    "capacity_fade_intercept": capacity_fade_intercept,
    "avg_early_charge_time": avg_early_charge_time,
    "mean_temperature": mean_temperature,
    "min_internal_resistance": min_internal_resistance,
    "internal_resistance_change": internal_resistance_change,
}


def _cell_features(qdlin: np.ndarray, summary: pd.DataFrame, names: list[str]) -> dict[str, float]:
    dq = delta_q(qdlin)
    out: dict[str, float] = {}
    for name in names:
        fn = _FEATURE_FNS[name]
        val = fn(dq) if name in _DQ_FEATURES else fn(summary)
        # Match the reference implementation: non-finite features collapse to 0.
        out[name] = 0.0 if not np.isfinite(val) else val
    return out


def build_feature_table(feature_set: str, ds: Dataset | None = None) -> pd.DataFrame:
    """Build a per-cell feature table for one feature set.

    Returns a DataFrame indexed by ``cell_id`` with the feature columns plus ``cycle_life``,
    ``log_cycle_life`` (the regression target), and ``split``.
    """
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; choose from {list(FEATURE_SETS)}")
    ds = ds or load_processed()
    names = FEATURE_SETS[feature_set]
    cells = ds.cells.set_index("cell_id")

    rows = []
    for cid in cells.index:
        summary = ds.summary[ds.summary["cell_id"] == cid].sort_values("cycle")
        feats = _cell_features(ds.qdlin[cid], summary, names)
        feats["cell_id"] = cid
        rows.append(feats)

    df = pd.DataFrame(rows).set_index("cell_id")
    df["cycle_life"] = cells["cycle_life"]
    df["log_cycle_life"] = np.log(cells["cycle_life"])
    df["split"] = cells["split"]
    return df[names + ["cycle_life", "log_cycle_life", "split"]]


def build_all() -> dict[str, pd.DataFrame]:
    """Build and persist all three feature tables to ``data/processed/features_*.parquet``."""
    from src.config import PROCESSED_DIR

    ds = load_processed()
    tables = {}
    for name in FEATURE_SETS:
        table = build_feature_table(name, ds)
        path = PROCESSED_DIR / f"features_{name}.parquet"
        table.to_parquet(path)
        tables[name] = table
        print(
            f"{name:>9} model: {len(FEATURE_SETS[name])} features x {len(table)} cells -> {path.name}"
        )
    return tables


if __name__ == "__main__":
    build_all()
