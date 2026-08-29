"""Configuration file loader for the NMC-Hierarchical-Breast-Cancer-Classifier.

Loads YAML configuration files from configs/ and provides
accessor functions for experiment parameters. Supports bare names
(e.g. "local", "server") that resolve to configs/<name>.yaml,
as well as full file paths.
"""

import json
from pathlib import Path

import yaml

# code/utils/config_loader.py -> code/utils -> code -> NMC-Hierarchical-Breast-Cancer-Classifier.
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "local.yaml"

# Backward-compatible aliases for the old trial/production names.
_ALIASES = {
    "trial": "local",
    "production": "server",
}


def load_config(config_path=None):
    """Load a YAML configuration file.

    Supports YAML (.yaml, .yml) and JSON (.json) files via three
    calling conventions:
      - ``load_config()`` loads the default ``configs/local.yaml``.
      - ``load_config("server")`` resolves to ``configs/server.yaml``.
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

        # Support bare names: "server" -> configs/server.yaml.
        if not config_path.suffix and not config_path.exists():
            config_path = CONFIG_DIR / f"{config_path.name}.yaml"

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix == ".json":
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    # Store the source path for logging and reproducibility.
    config["_config_path"] = str(config_path)

    return config


def get_grids(config):
    """Extract hyperparameter grids from a loaded config.

    Returns the grids dict from config. Plateau ensemble pipelines
    (_pens) are not expected to have grid entries because they pool
    results from their base pipeline's inner CV instead.

    Args:
        config (dict): Loaded configuration dictionary.

    Returns:
        dict: Pipeline name to param_grid mapping.
    """
    return config.get("grids", {})


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


def merge_cli_overrides(config, args):
    """Merge CLI argument values into a config dictionary.

    Creates a new dictionary. Only ``None`` values from args are skipped;
    ``0``, ``False``, and ``""`` are valid overrides.

    Args:
        config (dict): Base configuration dictionary.
        args (dict or argparse.Namespace): CLI arguments. If a Namespace,
            it is converted via ``vars()``.

    Returns:
        dict: Merged configuration (new dict, no mutation of inputs).
    """
    if not isinstance(args, dict):
        args = vars(args)
    merged = dict(config)
    for key, value in args.items():
        if value is not None:
            merged[key] = value
    return merged
