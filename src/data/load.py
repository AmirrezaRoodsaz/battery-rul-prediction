"""Parse raw Severson ``.mat`` batch files into in-memory per-cell structures.

The ``.mat`` files are MATLAB v7.3, which is HDF5 under the hood, so we read them with
``h5py``. MATLAB stores structs-of-arrays as HDF5 *object references*: e.g. ``batch['cycles']``
is an (n_cells, 1) array of references; dereferencing one (``f[ref]``) yields that cell's
cycle group, whose fields are themselves arrays of references to per-cycle vectors. The
dereferencing pattern below mirrors the paper's official loader, modernized for current h5py
(``.value`` → ``[()]``).

We extract **selectively**: every cell's per-cycle *summary* scalars (small) plus the
linearly-interpolated discharge curve ``Qdlin`` for only the first ``max_qdlin_cycle`` cycles.
That is all the feature engineering needs, and it keeps memory + the processed artifact small
instead of loading ~8 GB of full per-cycle time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

# The discharge Q(V) curve is interpolated onto a fixed 1000-point voltage grid by the
# dataset authors (the "Vdlin" grid). ΔQ(V) features rely on this fixed length.
QDLIN_LEN = 1000

# Summary fields are per-cycle scalar series (one value per cycle of the cell's life).
_SUMMARY_FIELDS = ("IR", "QCharge", "QDischarge", "Tavg", "Tmin", "Tmax", "chargetime", "cycle")
# We rename to short, consistent keys used throughout the pipeline.
_SUMMARY_RENAME = {
    "IR": "IR",
    "QCharge": "QC",
    "QDischarge": "QD",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
    "cycle": "cycle",
}


@dataclass
class Cell:
    """One battery cell's parsed data (selective: summary + early-cycle Qdlin)."""

    cell_id: str
    batch: int
    cycle_life: float
    charge_policy: str
    summary: dict[str, np.ndarray]
    # qdlin[k] is the 1000-point discharge Q(V) curve for early cycle index (k+1).
    qdlin: np.ndarray = field(repr=False)  # shape (n_early_cycles, 1000)


def _deref_scalar(f: h5py.File, ref) -> float:
    return float(np.array(f[ref]).flatten()[0])


def _deref_vector(f: h5py.File, ref) -> np.ndarray:
    return np.array(f[ref]).flatten()


def _decode_policy(f: h5py.File, ref) -> str:
    """policy_readable is stored as UTF-16-ish bytes; every other byte is the ASCII char."""
    raw = np.array(f[ref]).tobytes()
    return raw[::2].decode(errors="ignore").strip("\x00").strip()


def parse_batch(mat_path: str | Path, batch: int, max_qdlin_cycle: int = 100) -> dict[str, Cell]:
    """Parse one ``.mat`` batch file into ``{cell_id: Cell}``.

    Parameters
    ----------
    mat_path: path to the raw ``.mat`` file.
    batch: batch number (1, 2, 3) — used to build cell ids like ``b1c0``.
    max_qdlin_cycle: keep Qdlin only for the first this-many cycles (features use <=100).
    """
    mat_path = Path(mat_path)
    cells: dict[str, Cell] = {}

    with h5py.File(mat_path, "r") as f:
        b = f["batch"]
        n_cells = b["summary"].shape[0]

        for i in range(n_cells):
            cell_id = f"b{batch}c{i}"

            cycle_life = _deref_scalar(f, b["cycle_life"][i, 0])
            policy = _decode_policy(f, b["policy_readable"][i, 0])

            summary_grp = f[b["summary"][i, 0]]
            summary = {
                _SUMMARY_RENAME[name]: np.array(summary_grp[name][0, :]).flatten()
                for name in _SUMMARY_FIELDS
            }

            # Early-cycle Qdlin curves (each ~1000 points). cycles['Qdlin'] is (n_cyc, 1) refs.
            # Row j is aligned to cycle index j so downstream code can index cycle 10/100
            # directly. A few cycles (notably the initial partial cycle, index 0) carry a
            # degenerate non-1000-length curve; we store those as NaN rather than dropping
            # them, which would silently shift every later cycle's index.
            cyc_grp = f[b["cycles"][i, 0]]
            n_cyc = cyc_grp["Qdlin"].shape[0]
            keep = min(max_qdlin_cycle, n_cyc)
            qdlin = np.full((keep, QDLIN_LEN), np.nan)
            for j in range(keep):
                v = _deref_vector(f, cyc_grp["Qdlin"][j, 0])
                if v.size == QDLIN_LEN:
                    qdlin[j] = v

            cells[cell_id] = Cell(
                cell_id=cell_id,
                batch=batch,
                cycle_life=cycle_life,
                charge_policy=policy,
                summary=summary,
                qdlin=qdlin,
            )

    return cells
