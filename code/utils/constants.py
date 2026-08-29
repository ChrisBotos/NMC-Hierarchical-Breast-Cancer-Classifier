"""Shared constants used across all pipeline scripts."""

# Subtype display colours (consistent with Nature Reviews Cancer palette).
SUBTYPE_COLORS = {"HER2+": "#E64B35", "HR+": "#4DBBD5", "Triple Neg": "#00A087"}

# Canonical ordering of subtypes in all plots and tables.
SUBTYPE_ORDER = ["HER2+", "HR+", "Triple Neg"]

# Human-readable labels for discrete copy-number states.
CN_LABELS = {-1: "Loss", 0: "Normal", 1: "Gain", 2: "Amplification"}

# Column names that hold genomic coordinates (not sample data).
GENOMIC_COLUMNS = ("Chromosome", "Start", "End", "Nclone")

# Global random seed for reproducibility.
RANDOM_SEED = 42

# Pipeline display names for figures and tables.
PIPELINE_LABELS = {
    "kw_nmc": "KW + NMC",
    "kw_rf": "KW + RF",
    "en_nmc": "EN + NMC",
    "en_rf": "EN + RF",
    "standalone_en": "Standalone EN",
    "nmc_ensemble": "NMC Ensemble",
    "kw_nmc_pens": "KW + NMC (plateau)",
    "standalone_en_pens": "EN (plateau)",
    "en_nmc_pens": "EN + NMC (plateau)",
    "nmc_pens_ensemble": "NMC Pens Ensemble",
    "en_nmc_pens_p50": "Plateau n=50",
    "en_nmc_pens_p80": "Plateau n=80",
    "kw_nmc_kens": "KW + NMC (k-ensemble)",
    "kw_nmc_kgrid": "KW + NMC (k-grid)",
}

# Pipeline colours (consistent palette for all variants).
PIPELINE_COLORS = {
    "kw_nmc": "#4DBBD5",
    "kw_rf": "#E64B35",
    "en_nmc": "#00A087",
    "en_rf": "#F39B7F",
    "standalone_en": "#B09C85",
    "nmc_ensemble": "#8491B4",
    "kw_nmc_pens": "#7E6148",
    "standalone_en_pens": "#DC0000",
    "en_nmc_pens": "#B65284",
    "nmc_pens_ensemble": "#3C5488",
    "en_nmc_pens_p50": "#E64B35",
    "en_nmc_pens_p80": "#00A087",
    "kw_nmc_kens": "#91D1C2",
    "kw_nmc_kgrid": "#F39B7F",
}

# Canonical ordering for all hierarchical pipeline variants.
PIPELINE_NAMES = (
    "kw_nmc",
    "en_nmc",
    "kw_rf",
    "en_rf",
    "standalone_en",
    "nmc_ensemble",
    "kw_nmc_pens",
    "standalone_en_pens",
    "en_nmc_pens",
    "nmc_pens_ensemble",
)

# Pipelines that use GridSearchCV for Stage 2 hyperparameter tuning.
GRIDSEARCH_PIPELINES = ("kw_nmc", "en_nmc", "kw_rf", "en_rf", "standalone_en")

# Post-hoc ensemble pipelines: map ensemble name to its component pipeline names.
# These pipelines read pre-computed fold results from their components and
# average the 3-class probabilities. They do not run their own fold loop.
POSTHOC_ENSEMBLE_COMPONENTS = {
    "nmc_ensemble": ("kw_nmc", "en_nmc"),
    "nmc_pens_ensemble": ("kw_nmc_pens", "en_nmc_pens"),
}

# Plateau ensemble pipelines: map ensemble name to its base pipeline name.
PLATEAU_ENSEMBLE_BASE = {
    "kw_nmc_pens": "kw_nmc",
    "standalone_en_pens": "standalone_en",
    "en_nmc_pens": "en_nmc",
}

# Maximum number of models in a plateau ensemble (caps compute cost).
MAX_PLATEAU_SIZE = 15
