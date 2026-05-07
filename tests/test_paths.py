"""Tests for utils.paths module."""

import json
import shutil
from pathlib import Path

import pytest

from utils.paths import (
    CODE_DIR,
    CONFIGS_DIR,
    DATA_DIR,
    DATA_EXTERNAL_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DOCS_DIR,
    INTERESTING_RESULTS_DIR,
    NOTEBOOKS_DIR,
    PROJECT_DIR,
    REFERENCE_DOCS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
    _find_or_create_run_dir,
    get_run_dirs,
    get_run_dirs_no_replace,
    save_config,
)


class TestPathConstants:
    """Verify all path constants are correctly derived from __file__."""

    def test_project_dir_exists(self):
        """PROJECT_DIR should point to the actual project root."""
        assert PROJECT_DIR.exists()
        assert (PROJECT_DIR / "CLAUDE.md").exists() or (PROJECT_DIR / "code").exists()

    def test_code_dir_is_child_of_project(self):
        """CODE_DIR should be directly under PROJECT_DIR."""
        assert CODE_DIR == PROJECT_DIR / "code"

    def test_data_dir_structure(self):
        """DATA_DIR and its subdirectories should be correctly defined."""
        assert DATA_DIR == PROJECT_DIR / "data"
        assert DATA_RAW_DIR == DATA_DIR / "raw"
        assert DATA_INTERIM_DIR == DATA_DIR / "interim"
        assert DATA_PROCESSED_DIR == DATA_DIR / "processed"
        assert DATA_EXTERNAL_DIR == DATA_DIR / "external"

    def test_results_dir(self):
        """RESULTS_DIR should point to results/."""
        assert RESULTS_DIR == PROJECT_DIR / "results"

    def test_configs_dir(self):
        """CONFIGS_DIR should point to configs/."""
        assert CONFIGS_DIR == PROJECT_DIR / "configs"

    def test_other_dirs(self):
        """All other directory constants should be correctly derived."""
        assert INTERESTING_RESULTS_DIR == PROJECT_DIR / "interesting_results"
        assert DOCS_DIR == PROJECT_DIR / "docs"
        assert REPORTS_DIR == PROJECT_DIR / "reports"
        assert SCRIPTS_DIR == PROJECT_DIR / "scripts"
        assert NOTEBOOKS_DIR == PROJECT_DIR / "notebooks"
        assert REFERENCE_DOCS_DIR == PROJECT_DIR / "reference_documents"
        assert TESTS_DIR == PROJECT_DIR / "tests"


class TestFindOrCreateRunDir:
    """Tests for _find_or_create_run_dir."""

    def test_creates_new_run_dir(self, tmp_results, monkeypatch):
        """Should create a new date-prefixed directory when none exists."""
        monkeypatch.setattr("utils.paths.RESULTS_DIR", tmp_results)
        run_dir = _find_or_create_run_dir("test_run")
        assert run_dir.exists()
        assert run_dir.name.endswith("_test_run")

    def test_reuses_existing_run_dir(self, tmp_results, monkeypatch):
        """Should reuse an existing matching directory."""
        monkeypatch.setattr("utils.paths.RESULTS_DIR", tmp_results)
        existing = tmp_results / "2026-01-01_test_run"
        existing.mkdir()
        run_dir = _find_or_create_run_dir("test_run")
        assert run_dir == existing


class TestGetRunDirs:
    """Tests for get_run_dirs."""

    def test_creates_phase_subdirs(self, tmp_results, monkeypatch):
        """Should create data/, figures/, logs/ under the phase directory."""
        monkeypatch.setattr("utils.paths.RESULTS_DIR", tmp_results)
        fig_dir, data_dir, log_dir, run_dir = get_run_dirs("my_run", "eda")
        assert fig_dir.exists()
        assert data_dir.exists()
        assert log_dir.exists()
        assert fig_dir.name == "figures"
        assert data_dir.name == "data"
        assert log_dir.name == "logs"

    def test_phase_replacement_deletes_existing(self, tmp_results, monkeypatch):
        """Should delete and recreate phase dir on re-run."""
        monkeypatch.setattr("utils.paths.RESULTS_DIR", tmp_results)
        _, data_dir1, _, _ = get_run_dirs("my_run", "eda")
        marker = data_dir1 / "old_file.txt"
        marker.write_text("old")
        _, data_dir2, _, _ = get_run_dirs("my_run", "eda")
        assert not (data_dir2 / "old_file.txt").exists()


class TestGetRunDirsNoReplace:
    """Tests for get_run_dirs_no_replace."""

    def test_preserves_existing_content(self, tmp_results, monkeypatch):
        """Should not delete existing phase directory content."""
        monkeypatch.setattr("utils.paths.RESULTS_DIR", tmp_results)
        _, data_dir1, _, _ = get_run_dirs_no_replace("my_run", "cv")
        marker = data_dir1 / "keep_me.txt"
        marker.write_text("important")
        _, data_dir2, _, _ = get_run_dirs_no_replace("my_run", "cv")
        assert (data_dir2 / "keep_me.txt").exists()


class TestSaveConfig:
    """Tests for save_config."""

    def test_creates_config_json(self, tmp_path):
        """Should create config.json with script entry."""
        save_config(tmp_path, "test_script", param1="value1")
        config_path = tmp_path / "config.json"
        assert config_path.exists()
        config = json.loads(config_path.read_text())
        assert "test_script" in config
        assert config["test_script"]["param1"] == "value1"
        assert "timestamp" in config["test_script"]

    def test_merges_multiple_scripts(self, tmp_path):
        """Should merge entries from multiple scripts."""
        save_config(tmp_path, "script_a", x=1)
        save_config(tmp_path, "script_b", y=2)
        config = json.loads((tmp_path / "config.json").read_text())
        assert "script_a" in config
        assert "script_b" in config
        assert config["script_a"]["x"] == 1
        assert config["script_b"]["y"] == 2
