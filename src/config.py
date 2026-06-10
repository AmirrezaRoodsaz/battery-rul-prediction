"""Central configuration: paths, constants, and the global random seed.

Keeping these in one place means every stage of the pipeline (download → process →
features → train → evaluate) agrees on where data lives and how randomness is seeded —
which is what makes the project reproducible from a clean clone.
"""

from __future__ import annotations

from pathlib import Path

# --- Reproducibility ---------------------------------------------------------------
# A single seed used everywhere we split, shuffle, or train. Fixing it means a clone
# of this repo reproduces the same numbers. (Severson defines its own fixed split too;
# this seed governs only model-internal randomness and within-train CV.)
RANDOM_SEED: int = 42

# --- Project paths -----------------------------------------------------------------
# Resolved relative to this file so the code works regardless of the caller's CWD.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# --- Domain constants (Severson dataset) -------------------------------------------
NOMINAL_CAPACITY_AH: float = 1.1  # A123 APR18650M1A nominal capacity
EOL_FRACTION: float = 0.80  # End-of-Life = 80 % of nominal ...
EOL_CAPACITY_AH: float = NOMINAL_CAPACITY_AH * EOL_FRACTION  # = 0.88 Ah

# Cycles used for early-life feature extraction. The headline ΔQ(V) feature compares
# the cycle-100 and cycle-10 discharge voltage–capacity curves.
CYCLE_EARLY: int = 10
CYCLE_LATE: int = 100


def ensure_dirs() -> None:
    """Create the data/model/report directories if they don't yet exist."""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
