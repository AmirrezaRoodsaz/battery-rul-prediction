"""Plotting functions for EDA and model evaluation.

Each function takes already-loaded data and an optional Matplotlib ``Axes`` so notebooks can
compose figures, while ``make_eda_figures`` renders the standalone PNGs the README links to.
Keeping plotting here (not in notebooks) means the figures are reproducible from a script.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

from src.config import CYCLE_EARLY, CYCLE_LATE, EOL_CAPACITY_AH, FIGURES_DIR
from src.data.dataset import Dataset, load_processed

# A single place for the "color cells by their cycle life" idea used across plots.
_CMAP = "viridis"


def plot_capacity_fade(ds: Dataset, ax: Axes | None = None, max_cycle: int = 1000) -> Axes:
    """Discharge-capacity fade curves for every cell, colored by cycle life.

    Physically: each line is one cell losing capacity as it ages. The horizontal line marks
    the 0.88 Ah (80 % of nominal) End-of-Life threshold that defines the prediction target.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    life = ds.cells.set_index("cell_id")["cycle_life"]
    norm = LogNorm(vmin=life.min(), vmax=life.max())
    cmap = plt.get_cmap(_CMAP)
    for cid, grp in ds.summary.groupby("cell_id"):
        grp = grp[grp["cycle"] <= max_cycle]
        ax.plot(grp["cycle"], grp["QD"], color=cmap(norm(life[cid])), lw=0.6, alpha=0.7)
    ax.axhline(EOL_CAPACITY_AH, color="crimson", ls="--", lw=1.2, label="End-of-Life (0.88 Ah)")
    ax.set_xlabel("Cycle number")
    ax.set_ylabel("Discharge capacity (Ah)")
    ax.set_title("Capacity fade per cell")
    ax.set_ylim(0.85, 1.1)
    ax.legend(loc="lower left", fontsize=8)
    _add_colorbar(ax, norm, "Cycle life")
    return ax


def plot_cycle_life_distribution(ds: Dataset, ax: Axes | None = None) -> Axes:
    """Histogram of cycle life by split — shows the heavy right tail that motivates log-space."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    colors = {"train": "#4C72B0", "test": "#DD8452", "secondary_test": "#55A868"}
    bins = np.linspace(ds.cells["cycle_life"].min(), ds.cells["cycle_life"].max(), 25)
    for split, color in colors.items():
        vals = ds.cells.loc[ds.cells["split"] == split, "cycle_life"]
        ax.hist(vals, bins=bins, alpha=0.7, label=f"{split} (n={len(vals)})", color=color)
    ax.set_xlabel("Cycle life (cycles to EOL)")
    ax.set_ylabel("Number of cells")
    ax.set_title("Cycle-life distribution by split")
    ax.legend(fontsize=8)
    return ax


def _delta_q(ds: Dataset, cid: str) -> np.ndarray:
    """ΔQ(V) = Q_discharge(cycle 100) − Q_discharge(cycle 10) on the fixed voltage grid.

    This is the paper's core signal: even before capacity fade is visible, the *shape* of how
    the discharge curve shifts between cycle 10 and 100 encodes degradation mode information.
    Cycle n is row n-1 (rows are 0-indexed by cycle).
    """
    q = ds.qdlin[cid]
    return q[CYCLE_LATE - 1] - q[CYCLE_EARLY - 1]


def plot_dqv_curves(ds: Dataset, ax: Axes | None = None) -> Axes:
    """ΔQ(V) curves for every cell, colored by cycle life — the headline feature, visualized."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    life = ds.cells.set_index("cell_id")["cycle_life"]
    norm = LogNorm(vmin=life.min(), vmax=life.max())
    cmap = plt.get_cmap(_CMAP)
    for cid in ds.qdlin:
        dq = _delta_q(ds, cid)
        if np.isnan(dq).any():
            continue
        ax.plot(ds.vdlin, dq, color=cmap(norm(life[cid])), lw=0.6, alpha=0.7)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(r"$\Delta Q_{100-10}(V)$  (Ah)")
    ax.set_title("Discharge-curve change between cycle 10 and 100")
    _add_colorbar(ax, norm, "Cycle life")
    return ax


def plot_variance_vs_life(ds: Dataset, ax: Axes | None = None) -> Axes:
    """The canonical relationship: log10 var(ΔQ(V)) vs log10 cycle life.

    A tight negative correlation here is *why* a single feature predicts cycle life so well —
    this plot is the visual justification for the variance model.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    rows = []
    for cid in ds.qdlin:
        dq = _delta_q(ds, cid)
        if np.isnan(dq).any():
            continue
        rows.append(
            (np.log10(np.var(dq)), np.log10(ds.cells.set_index("cell_id").loc[cid, "cycle_life"]))
        )
    x, y = np.array(rows).T
    corr = np.corrcoef(x, y)[0, 1]
    ax.scatter(x, y, s=18, alpha=0.7, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{var}\,\Delta Q_{100-10}(V)$")
    ax.set_ylabel(r"$\log_{10}$ cycle life")
    ax.set_title(f"Variance feature vs cycle life (Pearson r = {corr:.2f})")
    return ax


def _add_colorbar(ax: Axes, norm: Normalize, label: str) -> None:
    sm = ScalarMappable(norm=norm, cmap=_CMAP)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax, label=label)


def make_eda_figures(ds: Dataset | None = None, out_dir: Path = FIGURES_DIR) -> list[Path]:
    """Render the four EDA figures to ``reports/figures/`` and return their paths."""
    ds = ds or load_processed()
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {
        "eda_capacity_fade.png": plot_capacity_fade,
        "eda_cycle_life_distribution.png": plot_cycle_life_distribution,
        "eda_dqv_curves.png": plot_dqv_curves,
        "eda_variance_vs_life.png": plot_variance_vs_life,
    }
    paths = []
    for name, fn in specs.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        fn(ds, ax=ax)
        fig.tight_layout()
        p = out_dir / name
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


if __name__ == "__main__":
    for p in make_eda_figures():
        print(f"wrote {p}")
