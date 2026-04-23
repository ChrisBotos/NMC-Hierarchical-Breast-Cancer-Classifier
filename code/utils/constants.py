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
}

# Pipeline colours (consistent 4-colour palette).
PIPELINE_COLORS = {
    "kw_nmc": "#4DBBD5",
    "kw_rf": "#E64B35",
    "en_nmc": "#00A087",
    "en_rf": "#F39B7F",
}
