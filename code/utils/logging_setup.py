"""Logging factory for dual rich-console + file output."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from utils.paths import PROJECT_DIR


def setup_logging(script_name, tag=""):
    """Configure root logger with a rich console handler and a file handler.

    Clears any pre-existing handlers so the function can be called more
    than once in the same process (e.g. when the orchestrator re-imports
    a module).

    Args:
        script_name (str): Base name used for the log file, e.g.
            "data_exploration_phase".
        tag (str): Optional suffix appended to the log file name.

    Returns:
        tuple[logging.Logger, Console]: (logger, rich console).
    """
    log_dir = PROJECT_DIR / "results" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    log_file = log_dir / f"{script_name}{suffix}.log"

    console = Console()

    # Remove any existing handlers to allow re-initialisation.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, show_time=False, show_path=False),
            logging.FileHandler(log_file, mode="w"),
        ],
    )
    logger = logging.getLogger(script_name)
    return logger, console
