"""Tests for utils.config_loader module."""

import json
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from utils.config_loader import load_config, merge_cli_overrides


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_yaml_file(self, tmp_path):
        """Should load a YAML config file correctly."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"key": "value", "number": 42}))
        config = load_config(str(config_file))
        assert config["key"] == "value"
        assert config["number"] == 42
        assert "_config_path" in config

    def test_load_json_file(self, tmp_path):
        """Should load a JSON config file correctly."""
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps({"key": "value", "number": 42}))
        config = load_config(str(config_file))
        assert config["key"] == "value"
        assert config["number"] == 42

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("/nonexistent/path/to/config.yaml")

    def test_default_config_loads(self):
        """Default config (local.yaml) should load without error."""
        config = load_config()
        assert isinstance(config, dict)
        assert "_config_path" in config


class TestMergeCliOverrides:
    """Tests for merge_cli_overrides."""

    def test_none_values_skipped(self):
        """None values from args should not overwrite config."""
        config = {"a": 1, "b": 2}
        args = {"a": None, "b": 99}
        merged = merge_cli_overrides(config, args)
        assert merged["a"] == 1
        assert merged["b"] == 99

    def test_false_and_zero_are_valid_overrides(self):
        """0, False, and empty string should override config values."""
        config = {"flag": True, "count": 10, "name": "old"}
        args = {"flag": False, "count": 0, "name": ""}
        merged = merge_cli_overrides(config, args)
        assert merged["flag"] is False
        assert merged["count"] == 0
        assert merged["name"] == ""

    def test_does_not_mutate_original(self):
        """Should create a new dict, not mutate the input config."""
        config = {"a": 1}
        args = {"a": 2}
        merged = merge_cli_overrides(config, args)
        assert config["a"] == 1
        assert merged["a"] == 2

    def test_accepts_namespace(self):
        """Should accept argparse.Namespace as args."""
        config = {"name": "old"}
        args = Namespace(name="new", verbose=None)
        merged = merge_cli_overrides(config, args)
        assert merged["name"] == "new"
        assert "verbose" not in merged

    def test_adds_new_keys(self):
        """Args keys not in config should be added."""
        config = {"a": 1}
        args = {"b": 2}
        merged = merge_cli_overrides(config, args)
        assert merged["a"] == 1
        assert merged["b"] == 2
