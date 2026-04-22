"""Pipeline factory for the 2x2 experimental design.

Builds sklearn Pipelines ready for GridSearchCV. Hyperparameter grids,
repeat counts, and fixed pipeline parameters are loaded from YAML
config files in config_files/ via :mod:`utils.config_loader`.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.cv_components import (
    ElasticNetSelector,
    KruskalWallisSelector,
    NearestCentroidWithProba,
)

"""Pipeline Names"""

PIPELINE_NAMES = ("kw_nmc", "kw_rf", "en_nmc", "en_rf")


def build_pipeline(pipeline_name, random_state=42, config=None):
    """Construct an sklearn Pipeline for the given pipeline name.

    Pipelines follow the project's 2x2 design:
    - KW pipelines: KruskalWallisSelector -> classifier.
    - EN pipelines: StandardScaler -> ElasticNetSelector -> classifier.
    - NMC pipelines use NearestCentroidWithProba (adds predict_proba for
      AUROC) and include StandardScaler before the selector.
    - RF pipelines tune max_features and min_samples_leaf via the param
      grid. class_weight is set for consistency with the Elastic Net
      selector. n_estimators and class_weight are read from the config
      file (defaulting to 500 and "balanced" respectively).

    Args:
        pipeline_name (str): One of 'kw_nmc', 'kw_rf', 'en_nmc', 'en_rf'.
        random_state (int): Random seed for stochastic components (RF,
            ElasticNet solver).
        config (dict or None): Loaded configuration dictionary. If None,
            uses default values for fixed pipeline parameters.

    Returns:
        Pipeline: A configured sklearn Pipeline.

    Raises:
        ValueError: If pipeline_name is not recognised.
    """
    # Read fixed pipeline parameters from config if available.
    rf_n_estimators = 500
    rf_class_weight = "balanced"
    if config and "pipelines" in config:
        rf_n_estimators = config["pipelines"].get("rf_n_estimators", 500)
        rf_class_weight = config["pipelines"].get(
            "rf_class_weight", "balanced",
        )

    if pipeline_name == "kw_nmc":
        # NMC uses Euclidean distance, so scale before KW selection.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", KruskalWallisSelector()),
            ("clf", NearestCentroidWithProba()),
        ])

    if pipeline_name == "kw_rf":
        # RF is scale-invariant; no scaler needed.
        return Pipeline([
            ("selector", KruskalWallisSelector()),
            ("clf", RandomForestClassifier(
                n_estimators=rf_n_estimators,
                class_weight=rf_class_weight,
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    if pipeline_name == "en_nmc":
        # EN needs scaled input; NMC also benefits from scaling.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", ElasticNetSelector(random_state=random_state)),
            ("clf", NearestCentroidWithProba()),
        ])

    if pipeline_name == "en_rf":
        # EN needs scaled input; RF is scale-invariant but it does not hurt.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", ElasticNetSelector(random_state=random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=rf_n_estimators,
                class_weight=rf_class_weight,
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    raise ValueError(
        f"Unknown pipeline '{pipeline_name}'. "
        f"Choose from: {PIPELINE_NAMES}."
    )
