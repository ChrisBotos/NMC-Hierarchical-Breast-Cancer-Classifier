"""Shared utilities for the TB-Project pipeline.

Re-exports the most commonly used names for convenient access::

    from utils import SUBTYPE_COLORS, get_sample_columns, setup_logging
"""

from utils.constants import (
    CN_LABELS,
    GENOMIC_COLUMNS,
    RANDOM_SEED,
    SUBTYPE_COLORS,
    SUBTYPE_ORDER,
)
from utils.data_helpers import get_sample_columns, get_sample_matrix, load_gene_map
from utils.logging_setup import setup_logging
from utils.paths import CODE_DIR, DATA_DIR, PROJECT_DIR, get_phase_dirs
from utils.plotting import apply_plot_style
from utils.statistics import bonferroni_threshold, kruskal_wallis_per_region
from utils.cv_components import (
    ElasticNetSelector,
    KruskalWallisSelector,
    NearestCentroidWithProba,
)
from utils.cv_config import (
    PIPELINE_NAMES,
    PRODUCTION_GRIDS,
    TRIAL_GRIDS,
    build_pipeline,
)

__all__ = [
    "CN_LABELS",
    "CODE_DIR",
    "DATA_DIR",
    "ElasticNetSelector",
    "GENOMIC_COLUMNS",
    "KruskalWallisSelector",
    "NearestCentroidWithProba",
    "PIPELINE_NAMES",
    "PRODUCTION_GRIDS",
    "PROJECT_DIR",
    "RANDOM_SEED",
    "SUBTYPE_COLORS",
    "SUBTYPE_ORDER",
    "TRIAL_GRIDS",
    "apply_plot_style",
    "bonferroni_threshold",
    "build_pipeline",
    "get_phase_dirs",
    "get_sample_columns",
    "get_sample_matrix",
    "kruskal_wallis_per_region",
    "load_gene_map",
    "setup_logging",
]
