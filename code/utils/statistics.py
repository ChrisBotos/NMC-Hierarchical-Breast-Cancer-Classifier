"""Statistical testing utilities for copy-number data."""

import numpy as np
import pandas as pd
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


def format_p_value(p_corrected):
    """Format a Bonferroni-corrected p-value for display on plots.

    Args:
        p_corrected (float): Bonferroni-corrected p-value.

    Returns:
        str: Formatted string, e.g. "p < 0.001", "n.s.", or "p = 0.042".
    """
    if p_corrected < 0.001:
        return "p < 0.001"
    if p_corrected >= 1.0:
        return "n.s."
    return f"p = {p_corrected:.3f}"


def apply_bonferroni(p_value, n_comparisons):
    """Apply Bonferroni correction to a single p-value.

    Args:
        p_value (float): Raw p-value.
        n_comparisons (int): Number of comparisons.

    Returns:
        float: Corrected p-value, capped at 1.0.
    """
    return min(p_value * n_comparisons, 1.0)


def pairwise_wilcoxon(wide_df, groups, log):
    """Friedman test followed by pairwise Wilcoxon signed-rank tests.

    Performs a Friedman test across all groups. If significant (p < 0.05),
    runs pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Args:
        wide_df (pd.DataFrame): Wide-format DataFrame where rows are
            repeated measures (e.g. repeats) and columns are groups
            (e.g. pipeline names) containing the metric values.
        groups (list[str]): Column names to compare (must be present
            in wide_df).
        log (logging.Logger): Logger instance.

    Returns:
        tuple: (friedman_stat, friedman_p, pairwise_df) where pairwise_df
            has columns: pipeline_a, pipeline_b, statistic, p_value,
            p_corrected, significant. Returns None for pairwise_df if
            Friedman test is not significant.
    """
    if len(groups) < 2:
        log.warning("Fewer than 2 groups; skipping statistical tests.")
        return None, None, None

    # Drop rows with missing data.
    n_before = len(wide_df)
    clean = wide_df[groups].dropna()
    n_after = len(clean)
    if n_after < n_before:
        log.info(
            "Dropped %d incomplete rows (%d -> %d) for paired tests.",
            n_before - n_after, n_before, n_after,
        )
    if n_after < 3:
        log.warning("Fewer than 3 complete rows; skipping statistical tests.")
        return None, None, None

    # Friedman test.
    samples = [clean[g].values for g in groups]
    friedman_stat, friedman_p = stats.friedmanchisquare(*samples)
    log.info("Friedman test: chi2=%.4f, p=%.6f", friedman_stat, friedman_p)

    if friedman_p >= 0.05:
        log.info("Friedman test not significant (p >= 0.05); skipping pairwise tests.")
        return friedman_stat, friedman_p, None

    # Pairwise Wilcoxon signed-rank tests with Bonferroni correction.
    n_comparisons = len(groups) * (len(groups) - 1) // 2
    pairwise_rows = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a, b = groups[i], groups[j]
            stat_val, p_val = stats.wilcoxon(clean[a].values, clean[b].values)
            p_corrected = apply_bonferroni(p_val, n_comparisons)
            pairwise_rows.append({
                "pipeline_a": a,
                "pipeline_b": b,
                "statistic": round(stat_val, 4),
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            })
            log.info(
                "  %s vs %s: W=%.1f, p=%.6f, p_corrected=%.6f %s",
                a, b, stat_val, p_val, p_corrected,
                "*" if p_corrected < 0.05 else "",
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    return friedman_stat, friedman_p, pairwise_df


def nadeau_bengio_test(pivot_df, pipeline_names, k, n_samples, log):
    """Pairwise Nadeau-Bengio corrected resampled t-tests.

    The Nadeau-Bengio correction accounts for the non-independence of
    test sets in repeated k-fold CV, preventing inflation of the test
    statistic that occurs with naive paired t-tests on overlapping folds.

    Reference: Nadeau & Bengio (2003), "Inference for the
    Generalization Error", Machine Learning 52(3):239-281.

    Args:
        pivot_df (pd.DataFrame): Scores indexed by (repeat, outer_fold)
            with pipeline names as columns.
        pipeline_names (list[str]): Pipeline names to compare (must be
            columns in pivot_df).
        k (int): Number of outer CV folds.
        n_samples (int): Total number of samples in the dataset.
        log (logging.Logger): Logger instance.

    Returns:
        pd.DataFrame or None: Pairwise test results with columns:
            pipeline_a, pipeline_b, mean_diff, t_statistic, df, p_value,
            p_corrected, significant. None if fewer than 2 pipelines.
    """
    present = [p for p in pipeline_names if p in pivot_df.columns]
    if len(present) < 2:
        log.warning("Fewer than 2 pipelines; skipping Nadeau-Bengio tests.")
        return None

    n_comparisons = len(present) * (len(present) - 1) // 2
    n_test = n_samples // k
    n_train = n_samples - n_test
    r = pivot_df.index.get_level_values("repeat").nunique()
    kr = k * r
    correction = 1.0 / kr + n_test / n_train

    log.info(
        "Nadeau-Bengio tests: k=%d folds, r=%d repeats, "
        "n_test=%d, n_train=%d, correction_factor=%.4f",
        k, r, n_test, n_train, correction,
    )

    rows = []
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            a, b = present[i], present[j]
            # Compute pairwise differences at the fold level.
            d = (pivot_df[a] - pivot_df[b]).dropna().values
            d_bar = np.mean(d)
            s2 = np.var(d, ddof=1)

            # Corrected variance (Nadeau-Bengio equation 8).
            sigma2 = correction * s2
            if sigma2 == 0:
                t_stat = 0.0
                p_val = 1.0
            else:
                t_stat = d_bar / np.sqrt(sigma2)
                p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=kr - 1))

            p_corrected = apply_bonferroni(p_val, n_comparisons)
            rows.append({
                "pipeline_a": a,
                "pipeline_b": b,
                "mean_diff": round(d_bar, 6),
                "t_statistic": round(t_stat, 4),
                "df": kr - 1,
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
            })
            log.info(
                "  %s vs %s: diff=%.4f, t=%.3f, df=%d, p=%.6f, "
                "p_corrected=%.6f %s",
                a, b, d_bar, t_stat, kr - 1, p_val, p_corrected,
                "*" if p_corrected < 0.05 else "",
            )

    return pd.DataFrame(rows)


def bonferroni_threshold(n_tests, alpha=0.05):
    """Compute the Bonferroni-corrected significance threshold.

    Args:
        n_tests (int): Number of tests performed.
        alpha (float): Family-wise error rate.

    Returns:
        float: Corrected per-test threshold.
    """
    return alpha / n_tests
