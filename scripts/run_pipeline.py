"""End-to-end pipeline: raw → processed → features → train → evaluate.

A single entry point so the whole project reproduces with one command (`make pipeline`). Each
stage is also runnable on its own via the corresponding `make` target during iteration. The raw
download is intentionally NOT run here (it is multi-GB and rarely changes) — run `make download`
once first; this script checks the raw files exist and tells you what to do if they don't.
"""

from __future__ import annotations

import sys
import time

from src.config import RAW_DIR
from src.data.preprocess import RAW_FILES, build_processed
from src.features.build_features import build_all
from src.models.evaluate import evaluate
from src.models.train import train_all


def _check_raw() -> bool:
    missing = [f for f in RAW_FILES.values() if not (RAW_DIR / f).exists()]
    if missing:
        print("Raw data missing:", *(f"  - {m}" for m in missing), sep="\n")
        print("\nRun `make download` first (see data/README.md for the data source/license).")
        return False
    return True


def main() -> int:
    if not _check_raw():
        return 1
    stages = [
        ("Preprocess raw .mat → tidy parquet", build_processed),
        ("Build feature tables", build_all),
        ("Train models (CV on train, eval on test)", train_all),
        ("Evaluate → figures + results.md", evaluate),
    ]
    for i, (label, fn) in enumerate(stages, 1):
        print(f"\n=== [{i}/{len(stages)}] {label} ===")
        t = time.time()
        fn()
        print(f"--- done in {time.time() - t:.1f}s")
    print("\nPipeline complete. See reports/results.md and reports/figures/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
