"""Statistical testing utilities for copy-number data."""

import numpy as np
from rich.progress import track
from scipy import stats

from utils.constants import SUBTYPE_ORDER


def kruskal_wallis_per_region(cn_df, clinical_df, sample_cols):
    """Run Kruskal-Wallis H-test per genomic region across subtypes.

    Args:
        cn_df (pd.DataFrame): Copy-number data (regions as rows).
        clinical_df (pd.DataFrame): Clinical labels with Sample and Subgroup.
        sample_cols (list[str]): Sample column names.

    Returns:
        tuple[np.ndarray, np.ndarray]: (p-values, H-statistics) arrays,
            one entry per region.
    """
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))
    groups = {}
    for subtype in SUBTYPE_ORDER:
        groups[subtype] = [s for s in sample_cols if label_map.get(s) == subtype]

    pvals = []
    h_stats = []
    for idx in track(range(len(cn_df)), description="Kruskal-Wallis tests"):
        row = cn_df.iloc[idx]
        samples_by_group = [row[groups[s]].values.astype(float) for s in SUBTYPE_ORDER]
        h, p = stats.kruskal(*samples_by_group)
        pvals.append(p)
        h_stats.append(h)

    return np.array(pvals), np.array(h_stats)


def bonferroni_threshold(n_tests, alpha=0.05):
    """Compute the Bonferroni-corrected significance threshold.

    Args:
        n_tests (int): Number of tests performed.
        alpha (float): Family-wise error rate.

    Returns:
        float: Corrected per-test threshold.
    """
    return alpha / n_tests
