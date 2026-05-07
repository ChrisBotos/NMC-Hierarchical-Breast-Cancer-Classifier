"""Logging factory for dual rich-console + file output."""

import logging
import subprocess
import warnings
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from utils.paths import PROJECT_DIR

# Track installed handlers for cleanup on re-init.
_active_handlers = []


def _get_git_hash():
    """Return the short git hash of the current commit, or 'unknown'.

    Returns:
        str: Short git hash string.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_DIR,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def setup_logging(script_name, tag="", log_dir=None, console=None, verbose=False):
    """Configure logging with a rich console handler and a file handler.

    Clears any pre-existing handlers so the function can be called more
    than once in the same process (e.g. when the orchestrator re-imports
    a module).

    Args:
        script_name (str): Base name used for the log file, e.g.
            "data_exploration_phase".
        tag (str): Optional suffix appended to the log file name.
        log_dir (Path | None): Directory for the log file. When None,
            falls back to the default results/logs/ location with a
            warning.
        console (Console | None): Rich console instance to use for the
            console handler. A new one is created when None.
        verbose (bool): If True, sets log level to DEBUG instead of INFO.

    Returns:
        tuple[logging.Logger, Console]: (logger, rich console).
    """
    global _active_handlers

    if log_dir is None:
        log_dir = PROJECT_DIR / "results" / "logs"
        warnings.warn(
            f"log_dir not specified, falling back to {log_dir}",
            stacklevel=2,
        )
    else:
        log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    log_file = log_dir / f"{script_name}{suffix}.log"

    if console is None:
        console = Console()

    level = logging.DEBUG if verbose else logging.INFO

    # Clean up previously installed handlers.
    root = logging.getLogger()
    for handler in _active_handlers:
        root.removeHandler(handler)
    _active_handlers.clear()

    root.setLevel(level)

    # File handler with timestamps for complete logging.
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    file_handler.setLevel(level)
    root.addHandler(file_handler)
    _active_handlers.append(file_handler)

    # Rich console handler for formatted interactive output.
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    rich_handler.setLevel(level)
    root.addHandler(rich_handler)
    _active_handlers.append(rich_handler)

    # Use a named logger with propagation disabled to avoid double output.
    logger = logging.getLogger(script_name)
    logger.propagate = False
    logger.setLevel(level)

    # Attach the same handlers to the named logger.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(file_handler)
    logger.addHandler(rich_handler)

    # Log the git hash as the first message for traceability.
    git_hash = _get_git_hash()
    logger.info(f"Starting {script_name} (git: {git_hash}).")

    return logger, console
