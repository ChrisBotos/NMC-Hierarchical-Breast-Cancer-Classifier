"""Tests for utils.logging_setup module."""

import logging

import pytest

from utils.logging_setup import _get_git_hash, setup_logging


class TestGetGitHash:
    """Tests for _get_git_hash."""

    def test_returns_string(self):
        """Should return a non-empty string."""
        result = _get_git_hash()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_short_hash_or_unknown(self):
        """Should return a short hash (7-12 chars) or 'unknown'."""
        result = _get_git_hash()
        if result != "unknown":
            assert 4 <= len(result) <= 12


class TestSetupLogging:
    """Tests for setup_logging."""

    def test_returns_logger_and_console(self, tmp_path):
        """Should return a (logger, Console) tuple."""
        logger, console = setup_logging("test_script", log_dir=tmp_path)
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_script"

    def test_creates_log_file(self, tmp_path):
        """Should create a log file in log_dir."""
        setup_logging("test_script", log_dir=tmp_path)
        log_files = list(tmp_path.glob("*.log"))
        assert len(log_files) == 1
        assert log_files[0].name == "test_script.log"

    def test_tag_suffix(self, tmp_path):
        """Tag should be appended to log file name."""
        setup_logging("test_script", tag="merged", log_dir=tmp_path)
        log_files = list(tmp_path.glob("*.log"))
        assert log_files[0].name == "test_script_merged.log"

    def test_verbose_sets_debug_level(self, tmp_path):
        """Verbose=True should set logger level to DEBUG."""
        logger, _ = setup_logging("test_verbose", log_dir=tmp_path, verbose=True)
        assert logger.level == logging.DEBUG

    def test_default_level_is_info(self, tmp_path):
        """Default log level should be INFO."""
        logger, _ = setup_logging("test_info", log_dir=tmp_path)
        assert logger.level == logging.INFO

    def test_logger_propagation_disabled(self, tmp_path):
        """Named logger should have propagation disabled."""
        logger, _ = setup_logging("test_prop", log_dir=tmp_path)
        assert logger.propagate is False

    def test_first_message_contains_git_hash(self, tmp_path):
        """First log line should contain the git hash."""
        setup_logging("test_git", log_dir=tmp_path)
        log_content = (tmp_path / "test_git.log").read_text()
        assert "git:" in log_content

    def test_fallback_warns_when_no_log_dir(self):
        """Should issue a warning when log_dir is None."""
        with pytest.warns(UserWarning, match="log_dir not specified"):
            setup_logging("test_fallback")

    def test_reinit_cleans_handlers(self, tmp_path):
        """Calling setup_logging twice should not duplicate handlers."""
        setup_logging("test_reinit", log_dir=tmp_path)
        logger, _ = setup_logging("test_reinit", log_dir=tmp_path)
        assert len(logger.handlers) == 2
