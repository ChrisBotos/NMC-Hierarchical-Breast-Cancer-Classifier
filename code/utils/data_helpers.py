"""Data loading and column selection helpers."""

import pandas as pd

from utils.constants import GENOMIC_COLUMNS


def get_sample_columns(cn_df):
    """Return list of sample column names (excluding genomic coordinate columns).

    Args:
        cn_df (pd.DataFrame): Copy-number data with genomic and sample columns.

    Returns:
        list[str]: Sample column names.
    """
    return [c for c in cn_df.columns if c not in GENOMIC_COLUMNS]


def get_sample_matrix(cn_df):
    """Return samples-by-regions matrix (rows=samples, columns=regions).

    Args:
        cn_df (pd.DataFrame): Copy-number data.

    Returns:
        pd.DataFrame: Transposed sample matrix.
    """
    sample_cols = get_sample_columns(cn_df)
    return cn_df[sample_cols].T


def load_gene_map(data_dir):
    """Load and harmonise the basepair-to-gene mapping.

    Chromosome strings "X" and "Y" are converted to integers 23 and 24
    to match the CN data convention.

    Args:
        data_dir (pathlib.Path): Path to the data/ directory.

    Returns:
        pd.DataFrame: Gene map with integer Chromosome column.
    """
    gene_map_df = pd.read_csv(data_dir / "BasepairToGeneMap.tsv", sep="\t")
    gene_map_df["Chromosome"] = gene_map_df["Chromosome"].replace(
        {"X": "23", "Y": "24"},
    )
    gene_map_df["Chromosome"] = pd.to_numeric(
        gene_map_df["Chromosome"],
        errors="coerce",
    )
    gene_map_df = gene_map_df.dropna(subset=["Chromosome"])
    gene_map_df["Chromosome"] = gene_map_df["Chromosome"].astype(int)
    return gene_map_df
