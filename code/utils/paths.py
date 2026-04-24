"""Centralised path resolution for the TB-Project.

All paths are derived from the location of this file so scripts work
from any working directory.
"""

import fcntl
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


# code/utils/paths.py -> code/utils -> code -> TB-Project.
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"


def get_phase_dirs(phase_name, tag=""):
    """Create and return (FIG_DIR, OUT_DIR) for a pipeline phase.

    Directories are created on disk if they do not already exist.

    Args:
        phase_name (str): Base name, e.g. "preprocessing_phase".
        tag (str): Optional suffix appended after the phase name
            (e.g. "merged" -> "data_exploration_phase_merged").

    Returns:
        tuple[Path, Path]: (figures directory, data output directory).
    """
    suffix = f"_{tag}" if tag else ""
    fig_dir = PROJECT_DIR / "results" / "figures" / f"{phase_name}{suffix}"
    out_dir = PROJECT_DIR / "results" / "data" / f"{phase_name}{suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, out_dir


def _find_or_create_run_dir(run_name):
    """Locate an existing run directory or create a new date-prefixed one.

    Searches results/ for a directory matching *_<run_name>. If found,
    reuses it. Otherwise creates <YYYY-MM-DD>_<run_name>/.

    Args:
        run_name (str): The run name (without date prefix).

    Returns:
        Path: The run root directory.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Search for an existing directory ending with _<run_name>.
    pattern = f"*_{run_name}"
    matches = sorted(RESULTS_DIR.glob(pattern))
    if matches:
        return matches[-1]

    # Create a new run directory with today's date prefix.
    today = datetime.now().strftime("%Y-%m-%d")
    run_dir = RESULTS_DIR / f"{today}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_run_dirs(run_name, phase_name):
    """Resolve or create results/<date>_<run_name>/<phase>/{data,figures,logs}.

    Searches results/ for an existing *_<run_name>/ directory. If found,
    reuses it. If not, creates <today>_<run_name>/. If the <phase>/
    subdirectory already exists inside the run dir, it is deleted and
    recreated (phase replacement).

    Args:
        run_name (str): The run name (e.g. "default_run", "experiment_1").
        phase_name (str): The phase directory name (e.g. "eda", "preprocessing").

    Returns:
        tuple[Path, Path, Path, Path]: (fig_dir, data_dir, log_dir, run_dir)
            where run_dir is the run root for saving config.json.
    """
    run_dir = _find_or_create_run_dir(run_name)
    phase_dir = run_dir / phase_name

    # Phase replacement: delete existing phase directory and recreate.
    if phase_dir.exists():
        shutil.rmtree(phase_dir)

    fig_dir = phase_dir / "figures"
    data_dir = phase_dir / "data"
    log_dir = phase_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, data_dir, log_dir, run_dir


def get_run_dirs_no_replace(run_name, phase_name):
    """Like get_run_dirs but without phase replacement.

    Creates the phase subdirectories if they do not exist, but never
    deletes existing content. Safe for parallel jobs that write to the
    same phase directory concurrently (e.g. nested CV array tasks).

    Args:
        run_name (str): The run name (e.g. "default_run").
        phase_name (str): The phase directory name (e.g. "nested_cv_2x2").

    Returns:
        tuple[Path, Path, Path, Path]: (fig_dir, data_dir, log_dir, run_dir).
    """
    run_dir = _find_or_create_run_dir(run_name)
    phase_dir = run_dir / phase_name

    fig_dir = phase_dir / "figures"
    data_dir = phase_dir / "data"
    log_dir = phase_dir / "logs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    return fig_dir, data_dir, log_dir, run_dir


def save_config(run_dir, script_name, **kwargs):
    """Save or update a config.json snapshot in the run root directory.

    Merges new entries into an existing config.json if present. Each
    entry is keyed by script_name with a timestamp and any extra
    keyword arguments. Uses file locking to prevent corruption when
    multiple concurrent jobs write to the same config.json.

    Args:
        run_dir (Path): The run root directory.
        script_name (str): Name of the script saving the config.
        **kwargs: Arbitrary parameters to record (e.g. pipeline, repeat, config).
    """
    config_path = run_dir / "config.json"
    lock_path = run_dir / ".config.json.lock"

    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        config[script_name] = {
            "timestamp": datetime.now().isoformat(),
            "command": " ".join(sys.argv),
            **kwargs,
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
