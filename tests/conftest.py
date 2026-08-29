"""Shared pytest fixtures for NMC-Hierarchical-Breast-Cancer-Classifier tests."""

import sys
from pathlib import Path

import pytest

# Ensure code/ is importable.
_CODE_DIR = Path(__file__).resolve().parent.parent / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))


@pytest.fixture
def seed():
    """Deterministic seed for reproducibility."""
    return 42


@pytest.fixture
def project_dir():
    """Project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_config():
    """Minimal config dictionary for testing."""
    return {
        "cv": {"n_repeats": 3, "n_outer_folds": 5, "n_inner_folds": 5},
        "pipelines": {"names": ["kw_nmc", "kw_rf"]},
        "grids": {"kw_nmc": {"selector__k": [5, 10]}},
    }


@pytest.fixture
def tmp_results(tmp_path):
    """Temporary results directory for testing run management."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return results_dir
