"""Shared utilities for the TB-Project pipeline.

Re-exports the most commonly used names for convenient access::

    from utils import SUBTYPE_COLORS, get_sample_columns, setup_logging
"""

from utils.constants import (
    CN_LABELS,
    GENOMIC_COLUMNS,
    PIPELINE_COLORS,
    PIPELINE_LABELS,
    RANDOM_SEED,
    SUBTYPE_COLORS,
    SUBTYPE_ORDER,
)
from utils.data_helpers import get_sample_columns, get_sample_matrix, load_gene_map
from utils.logging_setup import setup_logging
from utils.paths import (
    CODE_DIR,
    DATA_DIR,
    PROJECT_DIR,
    RESULTS_DIR,
    get_phase_dirs,
    get_run_dirs,
    get_run_dirs_no_replace,
    save_config,
)
from utils.plotting import annotate_heatmap, apply_plot_style, draw_significance_brackets
from utils.statistics import (
    apply_bonferroni,
    bonferroni_threshold,
    format_p_value,
    kruskal_wallis_per_region,
    nadeau_bengio_test,
    pairwise_wilcoxon,
)
from utils.cv_components import (
    ElasticNetSelector,
    KruskalWallisSelector,
    NearestCentroidWithProba,
)
from utils.cv_config import (
    PIPELINE_NAMES,
    build_pipeline,
)
from utils.cv_io import (
    checkpoint_path,
    csv_path,
    load_checkpoint,
    load_cv_data,
    resolve_merged_input,
    save_checkpoint,
)

__all__ = [
    "CN_LABELS",
    "CODE_DIR",
    "DATA_DIR",
    "ElasticNetSelector",
    "GENOMIC_COLUMNS",
    "KruskalWallisSelector",
    "NearestCentroidWithProba",
    "PIPELINE_COLORS",
    "PIPELINE_LABELS",
    "PIPELINE_NAMES",
    "PROJECT_DIR",
    "RANDOM_SEED",
    "RESULTS_DIR",
    "SUBTYPE_COLORS",
    "SUBTYPE_ORDER",
    "annotate_heatmap",
    "apply_bonferroni",
    "apply_plot_style",
    "bonferroni_threshold",
    "build_pipeline",
    "checkpoint_path",
    "csv_path",
    "draw_significance_brackets",
    "format_p_value",
    "get_phase_dirs",
    "get_run_dirs",
    "get_run_dirs_no_replace",
    "get_sample_columns",
    "get_sample_matrix",
    "kruskal_wallis_per_region",
    "load_checkpoint",
    "load_cv_data",
    "load_gene_map",
    "nadeau_bengio_test",
    "pairwise_wilcoxon",
    "resolve_merged_input",
    "save_checkpoint",
    "save_config",
    "setup_logging",
]
