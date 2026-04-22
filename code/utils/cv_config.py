"""Pipeline factory and hyperparameter grid configurations.

Defines trial (small) and production (full) hyperparameter grids for
the four experimental pipelines, and a factory function that builds
sklearn Pipelines ready for GridSearchCV.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils.cv_components import ElasticNetSelector, KruskalWallisSelector

"""Pipeline Names"""

PIPELINE_NAMES = ("kw_nmc", "kw_rf", "en_nmc", "en_rf")

"""Trial Grids (Small, for Development)"""

TRIAL_GRIDS = {
    "kw_nmc": {"selector__k": [5, 20]},
    "kw_rf": {"selector__k": [5, 20]},
    "en_nmc": {
        "selector__C": [0.01, 0.1, 1.0, 10.0],
        "selector__l1_ratio": [0.1, 0.5, 0.9],
        "selector__top_k": [5, 20],
    },
    "en_rf": {
        "selector__C": [0.01, 0.1, 1.0, 10.0],
        "selector__l1_ratio": [0.1, 0.5, 0.9],
        "selector__top_k": [5, 20],
    },
}

"""Production Grids (Full, for Final Runs)"""

PRODUCTION_GRIDS = {
    "kw_nmc": {"selector__k": [5, 10, 15, 20, 30, 50, 75, 100]},
    "kw_rf": {"selector__k": [5, 10, 15, 20, 30, 50, 75, 100]},
    "en_nmc": {
        "selector__C": np.logspace(-3, 2, 10).tolist(),
        "selector__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "selector__top_k": [5, 10, 20, 50],
    },
    "en_rf": {
        "selector__C": np.logspace(-3, 2, 10).tolist(),
        "selector__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "selector__top_k": [5, 10, 20, 50],
    },
}

"""Default Repeat Count Per Config"""

TRIAL_REPEATS = 3
PRODUCTION_REPEATS = 10


def build_pipeline(pipeline_name, random_state=42):
    """Construct an sklearn Pipeline for the given pipeline name.

    Pipelines follow the project's 2x2 design:
    - KW pipelines: KruskalWallisSelector -> classifier.
    - EN pipelines: StandardScaler -> ElasticNetSelector -> classifier.
    - NMC pipelines include StandardScaler before the selector (NMC uses
      Euclidean distance and needs scaled features).
    - RF pipelines use fixed hyperparameters (500 trees, min_samples_leaf=2)
      per the research plan.

    Args:
        pipeline_name (str): One of 'kw_nmc', 'kw_rf', 'en_nmc', 'en_rf'.
        random_state (int): Random seed for stochastic components (RF,
            ElasticNet solver).

    Returns:
        Pipeline: A configured sklearn Pipeline.

    Raises:
        ValueError: If pipeline_name is not recognised.
    """
    if pipeline_name == "kw_nmc":
        # NMC uses Euclidean distance, so scale before KW selection.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", KruskalWallisSelector()),
            ("clf", NearestCentroid()),
        ])

    if pipeline_name == "kw_rf":
        # RF is scale-invariant; no scaler needed.
        return Pipeline([
            ("selector", KruskalWallisSelector()),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    if pipeline_name == "en_nmc":
        # EN needs scaled input; NMC also benefits from scaling.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", ElasticNetSelector(random_state=random_state)),
            ("clf", NearestCentroid()),
        ])

    if pipeline_name == "en_rf":
        # EN needs scaled input; RF is scale-invariant but it does not hurt.
        return Pipeline([
            ("scaler", StandardScaler()),
            ("selector", ElasticNetSelector(random_state=random_state)),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    raise ValueError(
        f"Unknown pipeline '{pipeline_name}'. "
        f"Choose from: {PIPELINE_NAMES}."
    )
