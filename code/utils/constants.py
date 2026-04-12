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
