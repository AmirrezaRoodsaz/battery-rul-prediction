"""Smoke tests for the central config — cheap sanity checks that fail loudly if the
domain constants or path wiring get broken. Real feature/loader tests arrive in later phases.
"""

from src import config


def test_eol_capacity_is_80_percent_of_nominal():
    # End-of-Life is defined by the dataset as 80 % of nominal (0.88 Ah). Guard the math.
    assert config.EOL_CAPACITY_AH == config.NOMINAL_CAPACITY_AH * config.EOL_FRACTION
    assert abs(config.EOL_CAPACITY_AH - 0.88) < 1e-9


def test_early_late_cycle_ordering():
    # The ΔQ(V) feature compares a late cycle (100) against an early one (10).
    assert config.CYCLE_EARLY < config.CYCLE_LATE


def test_paths_are_under_project_root():
    for path in (config.RAW_DIR, config.PROCESSED_DIR, config.MODELS_DIR, config.FIGURES_DIR):
        assert config.PROJECT_ROOT in path.parents
