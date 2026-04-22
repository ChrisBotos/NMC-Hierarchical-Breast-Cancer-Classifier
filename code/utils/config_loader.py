"""Configuration file loader for the TB-Project.

Loads YAML configuration files from config_files/ and provides
accessor functions for experiment parameters. Supports bare names
(e.g. "local", "server") that resolve to config_files/<name>.yaml,
as well as full file paths.
"""

from pathlib import Path

import yaml

# code/utils/config_loader.py -> code/utils -> code -> TB-Project.
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_DIR / "config_files"
DEFAULT_CONFIG = CONFIG_DIR / "local.yaml"

# Backward-compatible aliases for the old trial/production names.
_ALIASES = {
    "trial": "local",
    "production": "server",
}


def load_config(config_path=None):
    """Load a YAML configuration file.

    Supports three calling conventions:
      - ``load_config()`` loads the default ``config_files/local.yaml``.
      - ``load_config("server")`` resolves to ``config_files/server.yaml``.
      - ``load_config("/absolute/path/to/config.yaml")`` loads that file.

    The legacy names "trial" and "production" are mapped to "local" and
    "server" respectively for backward compatibility.

    Args:
        config_path (str or Path or None): Path or bare name of the
            config file. If None, loads the default local config.

    Returns:
        dict: The parsed configuration dictionary. Contains an extra
            ``_config_path`` key with the resolved source file path.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG
    else:
        config_path = str(config_path)

        # Apply legacy aliases.
        if config_path in _ALIASES:
            config_path = _ALIASES[config_path]

        config_path = Path(config_path)

        # Support bare names: "server" -> config_files/server.yaml.
        if not config_path.suffix and not config_path.exists():
            config_path = CONFIG_DIR / f"{config_path.name}.yaml"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Store the source path for logging and reproducibility.
    config["_config_path"] = str(config_path)

    return config


def get_grids(config):
    """Extract hyperparameter grids from a loaded config.

    Args:
        config (dict): Loaded configuration dictionary.

    Returns:
        dict: Pipeline name to param_grid mapping.
    """
    return config["grids"]


def get_pipeline_names(config):
    """Extract pipeline names from a loaded config.

    Args:
        config (dict): Loaded configuration dictionary.

    Returns:
        tuple: Pipeline name strings.
    """
    return tuple(config["pipelines"]["names"])


def get_n_repeats(config):
    """Extract the number of CV repeats from a loaded config.

    Args:
        config (dict): Loaded configuration dictionary.

    Returns:
        int: Number of repeats.
    """
    return config["cv"]["n_repeats"]
