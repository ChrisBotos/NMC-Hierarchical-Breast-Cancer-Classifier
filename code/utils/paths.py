"""Centralised path resolution for the TB-Project.

All paths are derived from the location of this file so scripts work
from any working directory.
"""

from pathlib import Path


# code/utils/paths.py -> code/utils -> code -> TB-Project.
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = CODE_DIR.parent
DATA_DIR = PROJECT_DIR / "data"


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
