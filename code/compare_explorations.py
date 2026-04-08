"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: compare_explorations.py.
Description:
    Standalone comparison of exploration analyses on raw (2834 regions) vs
    merged (273 consensus segments) data. Produces a single multi-panel
    figure showing PCA, t-SNE, and summary metrics side-by-side.

Usage:
    python3 code/compare_explorations.py

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, scipy, matplotlib, rich.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback
from rich.console import Console
from rich.logging import RichHandler
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

rich.traceback.install()

# ---------------------------------------------------------------------------
# Path setup — works from any working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MERGED_DIR = PROJECT_DIR / "results" / "data" / "preprocessing_phase"
FIG_DIR = PROJECT_DIR / "results" / "figures" / "compare_explorations"
LOG_DIR = PROJECT_DIR / "results" / "logs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging setup — dual output to rich console and log file.
# ---------------------------------------------------------------------------
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(console=console, show_time=False, show_path=False),
        logging.FileHandler(LOG_DIR / "compare_explorations.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global settings.
# ---------------------------------------------------------------------------
SUBTYPE_COLORS = {"HER2+": "#E64B35", "HR+": "#4DBBD5", "Triple Neg": "#00A087"}
SUBTYPE_ORDER = ["HER2+", "HR+", "Triple Neg"]
RANDOM_SEED = 42

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "font.family": "sans-serif",
})


"""Data Loading."""


def load_data():
    """Load raw and merged training data plus clinical labels.

    Returns:
        tuple: (raw_df, merged_df, clinical_df).

    Raises:
        FileNotFoundError: If merged data does not exist (run
            preprocessing_phase.py first).
    """
    raw_df = pd.read_csv(DATA_DIR / "Train_call.tsv", sep="\t")
    clinical_df = pd.read_csv(DATA_DIR / "Train_clinical.tsv", sep="\t")

    merged_path = MERGED_DIR / "train_merged.tsv"
    if not merged_path.exists():
        raise FileNotFoundError(
            f"Merged data not found at {merged_path}. "
            "Run preprocessing_phase.py first."
        )
    merged_df = pd.read_csv(merged_path, sep="\t")

    return raw_df, merged_df, clinical_df


def get_sample_columns(cn_df):
    """Return sample column names (excluding genomic coordinate columns).

    Args:
        cn_df (pd.DataFrame): Copy-number data.

    Returns:
        list[str]: Sample column names.
    """
    return [c for c in cn_df.columns
            if c not in ("Chromosome", "Start", "End", "Nclone")]


"""Analysis Functions."""


def run_pca(X, n_components=10):
    """Fit PCA and return transformed data and explained variance ratios.

    Args:
        X (np.ndarray): Samples-by-features matrix.
        n_components (int): Number of components.

    Returns:
        tuple: (X_pca, explained_variance_ratio).
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_


def run_tsne(X, perplexity=30):
    """Fit t-SNE and return 2D embedding.

    Args:
        X (np.ndarray): Samples-by-features matrix.
        perplexity (int): t-SNE perplexity parameter.

    Returns:
        np.ndarray: 2D embedding (n_samples, 2).
    """
    tsne = TSNE(
        n_components=2, perplexity=perplexity,
        random_state=RANDOM_SEED, max_iter=1000,
    )
    return tsne.fit_transform(X)


def compute_silhouette(X):
    """Compute silhouette score for k=3 Ward clustering.

    Args:
        X (np.ndarray): Samples-by-features matrix.

    Returns:
        float: Silhouette score.
    """
    agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = agg.fit_predict(X)
    return silhouette_score(X, labels)


def compute_bonferroni_significant(cn_df, clinical_df, sample_cols):
    """Count Bonferroni-significant features via Kruskal-Wallis test.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        clinical_df (pd.DataFrame): Clinical labels.
        sample_cols (list[str]): Sample column names.

    Returns:
        int: Number of features passing Bonferroni correction.
    """
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))
    groups = {}
    for subtype in SUBTYPE_ORDER:
        groups[subtype] = [s for s in sample_cols
                           if label_map.get(s) == subtype]

    pvals = []
    for idx in range(len(cn_df)):
        row = cn_df.iloc[idx]
        samples_by_group = [
            row[groups[s]].values.astype(float) for s in SUBTYPE_ORDER
        ]
        _, p = stats.kruskal(*samples_by_group)
        pvals.append(p)

    pvals = np.array(pvals)
    bonf_alpha = 0.05 / len(pvals)
    return int((pvals < bonf_alpha).sum())


def compute_adjacent_corr_median(cn_df, sample_cols):
    """Compute median Pearson r between adjacent same-chromosome regions.

    Args:
        cn_df (pd.DataFrame): Copy-number data.
        sample_cols (list[str]): Sample column names.

    Returns:
        float: Median adjacent correlation.
    """
    data = cn_df[sample_cols].values.astype(float)
    chroms = cn_df["Chromosome"].values

    correlations = []
    for i in range(len(cn_df) - 1):
        if chroms[i] == chroms[i + 1]:
            r = np.corrcoef(data[i], data[i + 1])[0, 1]
            if not np.isnan(r):
                correlations.append(r)

    return float(np.median(correlations)) if correlations else 0.0


"""Figure."""


def make_comparison_figure(raw_df, merged_df, clinical_df):
    """Create a single multi-panel figure comparing raw vs merged exploration.

    Layout (2 rows x 3 columns):
        Row 1: PCA scatter (raw), PCA scatter (merged), scree overlay.
        Row 2: t-SNE (raw), t-SNE (merged), summary metrics bar chart.

    Args:
        raw_df (pd.DataFrame): Raw copy-number data (2834 regions).
        merged_df (pd.DataFrame): Merged consensus data (273 segments).
        clinical_df (pd.DataFrame): Clinical labels.
    """
    sample_cols = get_sample_columns(raw_df)
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))
    labels = [label_map[s] for s in sample_cols]

    X_raw = raw_df[sample_cols].T.values
    X_merged = merged_df[sample_cols].T.values

    # --- Compute all metrics ---
    log.info("Computing PCA (raw) ...")
    pca_raw, var_raw = run_pca(X_raw)
    log.info("Computing PCA (merged) ...")
    pca_merged, var_merged = run_pca(X_merged)

    log.info("Computing t-SNE (raw, perplexity=30) ...")
    tsne_raw = run_tsne(X_raw, perplexity=30)
    log.info("Computing t-SNE (merged, perplexity=30) ...")
    tsne_merged = run_tsne(X_merged, perplexity=30)

    log.info("Computing silhouette scores ...")
    sil_raw = compute_silhouette(X_raw)
    sil_merged = compute_silhouette(X_merged)
    log.info("  Raw: %.3f, Merged: %.3f", sil_raw, sil_merged)

    log.info("Computing Kruskal-Wallis (raw) ...")
    kw_raw = compute_bonferroni_significant(raw_df, clinical_df, sample_cols)
    log.info("Computing Kruskal-Wallis (merged) ...")
    kw_merged = compute_bonferroni_significant(
        merged_df, clinical_df, sample_cols,
    )
    log.info("  Bonferroni-significant: raw=%d/%d, merged=%d/%d",
             kw_raw, len(raw_df), kw_merged, len(merged_df))

    log.info("Computing adjacent correlations ...")
    adj_raw = compute_adjacent_corr_median(raw_df, sample_cols)
    adj_merged = compute_adjacent_corr_median(merged_df, sample_cols)
    log.info("  Median adjacent r: raw=%.3f, merged=%.3f",
             adj_raw, adj_merged)

    # --- Build figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    # Helper to plot a subtype-colored scatter.
    def scatter_by_subtype(ax, X_2d, panel_label, subtitle):
        """Plot a 2D scatter colored by subtype with panel label.

        Args:
            ax (matplotlib.axes.Axes): Target axes.
            X_2d (np.ndarray): 2D coordinates (n_samples, 2).
            panel_label (str): Panel letter, e.g. "a".
            subtitle (str): Text annotation inside the panel.
        """
        for subtype in SUBTYPE_ORDER:
            mask = [l == subtype for l in labels]
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=SUBTYPE_COLORS[subtype], label=subtype,
                s=30, alpha=0.8, edgecolors="black", linewidths=0.2,
            )
        ax.text(0.02, 0.98, f"({panel_label}) {subtitle}",
                transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)

    # (a) PCA raw.
    scatter_by_subtype(axes[0, 0], pca_raw,
                       "a", f"Raw data ({len(raw_df)} regions)")
    axes[0, 0].set_xlabel(f"PC1 ({var_raw[0]*100:.1f}% variance)")
    axes[0, 0].set_ylabel(f"PC2 ({var_raw[1]*100:.1f}% variance)")

    # (b) PCA merged.
    scatter_by_subtype(axes[0, 1], pca_merged,
                       "b", f"Merged data ({len(merged_df)} segments)")
    axes[0, 1].set_xlabel(f"PC1 ({var_merged[0]*100:.1f}% variance)")
    axes[0, 1].set_ylabel(f"PC2 ({var_merged[1]*100:.1f}% variance)")
    axes[0, 1].legend(framealpha=0.9, loc="upper right")

    # (c) Scree overlay.
    ax = axes[0, 2]
    pcs = range(1, 11)
    ax.plot(pcs, np.cumsum(var_raw) * 100, "o-", color="#3C5488",
            markersize=4, label=f"Raw ({len(raw_df)} regions)")
    ax.plot(pcs, np.cumsum(var_merged) * 100, "s-", color="#00A087",
            markersize=4, label=f"Merged ({len(merged_df)} segments)")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_xticks(list(pcs))
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.98, "(c) Cumulative variance",
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    # (d) t-SNE raw.
    scatter_by_subtype(axes[1, 0], tsne_raw,
                       "d", f"Raw data (perplexity=30)")
    axes[1, 0].set_xlabel("t-SNE dimension 1 (arbitrary units)")
    axes[1, 0].set_ylabel("t-SNE dimension 2 (arbitrary units)")

    # (e) t-SNE merged.
    scatter_by_subtype(axes[1, 1], tsne_merged,
                       "e", f"Merged data (perplexity=30)")
    axes[1, 1].set_xlabel("t-SNE dimension 1 (arbitrary units)")
    axes[1, 1].set_ylabel("t-SNE dimension 2 (arbitrary units)")

    # (f) Summary metrics grouped bars.
    ax = axes[1, 2]
    metric_names = [
        "Silhouette\nscore",
        "Cumulative\nvariance\n(PC1-10, %)",
        "Median\nadjacent\ncorrelation",
    ]
    raw_vals = [sil_raw, np.sum(var_raw) * 100, adj_raw]
    merged_vals = [sil_merged, np.sum(var_merged) * 100, adj_merged]

    x = np.arange(len(metric_names))
    width = 0.3
    bars_raw = ax.bar(x - width / 2, raw_vals, width, color="#3C5488",
                      edgecolor="black", linewidth=0.3,
                      label=f"Raw ({len(raw_df)})")
    bars_merged = ax.bar(x + width / 2, merged_vals, width, color="#00A087",
                         edgecolor="black", linewidth=0.3,
                         label=f"Merged ({len(merged_df)})")

    # Add value labels on bars.
    for bar in bars_raw:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.2f}" if h < 1 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars_merged:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.2f}" if h < 1 else f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=7)
    ax.set_ylabel("Value")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.98, "(f) Summary metrics",
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    # Annotate Bonferroni-significant count below the summary panel.
    ax.text(0.5, -0.18,
            f"Bonferroni-significant features: "
            f"raw = {kw_raw}/{len(raw_df)}, "
            f"merged = {kw_merged}/{len(merged_df)}",
            transform=ax.transAxes, fontsize=8, ha="center",
            style="italic")

    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    fig.savefig(FIG_DIR / "01_raw_vs_merged_comparison.png")
    plt.close(fig)
    log.info("Saved: 01_raw_vs_merged_comparison.png")


"""Main."""


def main():
    """Run exploration comparison: raw vs merged data."""
    log.info("=" * 60)
    log.info("COMPARE EXPLORATIONS: RAW VS MERGED DATA")
    log.info("=" * 60)

    raw_df, merged_df, clinical_df = load_data()
    log.info("Raw data: %d regions x %d samples",
             len(raw_df), len(get_sample_columns(raw_df)))
    log.info("Merged data: %d segments x %d samples",
             len(merged_df), len(get_sample_columns(merged_df)))
    log.info("")

    make_comparison_figure(raw_df, merged_df, clinical_df)

    log.info("")
    log.info("=" * 60)
    log.info("COMPARISON COMPLETE.")
    log.info("  Figure: %s", FIG_DIR)
    log.info("  Log:    %s", LOG_DIR / "compare_explorations.log")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
