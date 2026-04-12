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
    Compare two data exploration runs side-by-side. Loads pre-computed
    PCA, t-SNE, and Kruskal-Wallis results from exploration output
    directories (produced by data_exploration_phase.py) and generates
    a single multi-panel comparison figure.

Usage:
    python3 code/compare_explorations.py --dir-a results/data/data_exploration_phase --dir-b results/data/data_exploration_phase_merged
    python3 code/compare_explorations.py --dir-a results/data/data_exploration_phase --dir-b results/data/data_exploration_phase_merged --label-a "Raw (2834)" --label-b "Merged (273)" --tag raw_vs_merged

Dependencies:
    Python >= 3.10.
    pandas, numpy, matplotlib, rich.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rich.traceback

from utils import (
    DATA_DIR,
    PROJECT_DIR,
    SUBTYPE_COLORS,
    SUBTYPE_ORDER,
    apply_plot_style,
    setup_logging,
)

rich.traceback.install()

# Initialise logging.
log, console = setup_logging("compare_explorations")
apply_plot_style(scale="compact")

# Files that must exist in each exploration output directory.
REQUIRED_FILES = [
    "pca_coordinates.tsv",
    "pca_variance_explained.tsv",
    "tsne_perp30.tsv",
    "kruskal_wallis_per_region.tsv",
]


"""Validation."""


def validate_directory(dir_path):
    """Check that an exploration output directory has all required files.

    Args:
        dir_path (Path): Path to a data_exploration_phase output directory.

    Returns:
        list[str]: List of missing file names (empty if all present).
    """
    missing = []
    for fname in REQUIRED_FILES:
        if not (dir_path / fname).exists():
            missing.append(fname)
    return missing


"""Data Loading."""


def load_exploration(dir_path):
    """Load pre-computed exploration results from an output directory.

    Args:
        dir_path (Path): Path to a data_exploration_phase output directory.

    Returns:
        dict: Keys are 'pca_coords', 'pca_variance', 'tsne', 'kw'.
    """
    pca_coords = pd.read_csv(dir_path / "pca_coordinates.tsv", sep="\t", index_col=0)
    pca_var = pd.read_csv(dir_path / "pca_variance_explained.tsv", sep="\t")
    tsne = pd.read_csv(dir_path / "tsne_perp30.tsv", sep="\t", index_col=0)
    kw = pd.read_csv(dir_path / "kruskal_wallis_per_region.tsv", sep="\t")

    return {
        "pca_coords": pca_coords,
        "pca_variance": pca_var,
        "tsne": tsne,
        "kw": kw,
    }


"""Figure."""


def make_comparison_figure(data_a, data_b, label_a, label_b, clinical_df, fig_path):
    """Create a multi-panel figure comparing two exploration runs.

    Layout (2 rows x 3 columns):
        Row 1: PCA scatter (A), PCA scatter (B), cumulative variance overlay.
        Row 2: t-SNE (A), t-SNE (B), summary metrics bar chart.

    Args:
        data_a (dict): Exploration results for dataset A.
        data_b (dict): Exploration results for dataset B.
        label_a (str): Display label for dataset A.
        label_b (str): Display label for dataset B.
        clinical_df (pd.DataFrame): Clinical labels.
        fig_path (Path): Output path for the figure.
    """
    label_map = dict(zip(clinical_df["Sample"], clinical_df["Subgroup"]))

    # Build subtype label arrays aligned to PCA/t-SNE sample order.
    samples_a = data_a["pca_coords"].index.tolist()
    samples_b = data_b["pca_coords"].index.tolist()
    labels_a = [label_map.get(s, "Unknown") for s in samples_a]
    labels_b = [label_map.get(s, "Unknown") for s in samples_b]

    # Extract arrays.
    pca_a = data_a["pca_coords"].values
    pca_b = data_b["pca_coords"].values
    var_a = data_a["pca_variance"]["variance_explained"].values
    var_b = data_b["pca_variance"]["variance_explained"].values
    tsne_a = data_a["tsne"].values
    tsne_b = data_b["tsne"].values
    n_features_a = len(data_a["kw"])
    n_features_b = len(data_b["kw"])
    bonf_a = int((data_a["kw"]["significant_bonferroni"]).sum())
    bonf_b = int((data_b["kw"]["significant_bonferroni"]).sum())

    # --- Build figure ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

    def scatter_by_subtype(ax, X_2d, sample_labels, panel_label, subtitle):
        """Plot a 2D scatter colored by subtype with panel label.

        Args:
            ax (matplotlib.axes.Axes): Target axes.
            X_2d (np.ndarray): 2D coordinates (n_samples, 2).
            sample_labels (list[str]): Subtype label per sample.
            panel_label (str): Panel letter, e.g. "a".
            subtitle (str): Text annotation inside the panel.
        """
        for subtype in SUBTYPE_ORDER:
            mask = [l == subtype for l in sample_labels]
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=SUBTYPE_COLORS[subtype], label=subtype,
                s=30, alpha=0.8, edgecolors="black", linewidths=0.2,
            )
        ax.text(0.02, 0.98, f"({panel_label}) {subtitle}",
                transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)

    # (a) PCA A.
    scatter_by_subtype(axes[0, 0], pca_a[:, :2], labels_a,
                       "a", label_a)
    axes[0, 0].set_xlabel(f"PC1 ({var_a[0]*100:.1f}% variance)")
    axes[0, 0].set_ylabel(f"PC2 ({var_a[1]*100:.1f}% variance)")

    # (b) PCA B.
    scatter_by_subtype(axes[0, 1], pca_b[:, :2], labels_b,
                       "b", label_b)
    axes[0, 1].set_xlabel(f"PC1 ({var_b[0]*100:.1f}% variance)")
    axes[0, 1].set_ylabel(f"PC2 ({var_b[1]*100:.1f}% variance)")
    axes[0, 1].legend(framealpha=0.9, loc="upper right")

    # (c) Cumulative variance overlay.
    ax = axes[0, 2]
    pcs = range(1, len(var_a) + 1)
    ax.plot(pcs, np.cumsum(var_a) * 100, "o-", color="#3C5488",
            markersize=4, label=label_a)
    ax.plot(pcs, np.cumsum(var_b) * 100, "s-", color="#00A087",
            markersize=4, label=label_b)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_xticks(list(pcs))
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.98, "(c) Cumulative variance",
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    # (d) t-SNE A.
    scatter_by_subtype(axes[1, 0], tsne_a, labels_a,
                       "d", f"{label_a} (perplexity=30)")
    axes[1, 0].set_xlabel("t-SNE dimension 1 (arbitrary units)")
    axes[1, 0].set_ylabel("t-SNE dimension 2 (arbitrary units)")

    # (e) t-SNE B.
    scatter_by_subtype(axes[1, 1], tsne_b, labels_b,
                       "e", f"{label_b} (perplexity=30)")
    axes[1, 1].set_xlabel("t-SNE dimension 1 (arbitrary units)")
    axes[1, 1].set_ylabel("t-SNE dimension 2 (arbitrary units)")

    # (f) Summary metrics.
    ax = axes[1, 2]
    metric_names = [
        "Cumulative\nvariance\n(PC1-10, %)",
        "Bonferroni-\nsignificant\n(%)",
    ]
    bonf_pct_a = (bonf_a / n_features_a * 100) if n_features_a > 0 else 0
    bonf_pct_b = (bonf_b / n_features_b * 100) if n_features_b > 0 else 0
    vals_a = [np.sum(var_a) * 100, bonf_pct_a]
    vals_b = [np.sum(var_b) * 100, bonf_pct_b]

    x = np.arange(len(metric_names))
    width = 0.3
    bars_a = ax.bar(x - width / 2, vals_a, width, color="#3C5488",
                    edgecolor="black", linewidth=0.3, label=label_a)
    bars_b = ax.bar(x + width / 2, vals_b, width, color="#00A087",
                    edgecolor="black", linewidth=0.3, label=label_b)

    for bar in bars_a:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)
    for bar in bars_b:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=7)
    ax.set_ylabel("Value (%)")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.02, 0.98, "(f) Summary metrics",
            transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top")

    # Annotate feature counts and raw Bonferroni numbers below the panel.
    ax.text(0.5, -0.18,
            f"Features: A = {n_features_a}, B = {n_features_b}. "
            f"Bonferroni-significant: A = {bonf_a}/{n_features_a}, "
            f"B = {bonf_b}/{n_features_b}.",
            transform=ax.transAxes, fontsize=8, ha="center",
            style="italic")

    fig.subplots_adjust(hspace=0.35, wspace=0.30)
    fig.savefig(fig_path)
    plt.close(fig)
    log.info("Saved: %s", fig_path)


"""Main."""


def main():
    """Run exploration comparison between two datasets."""
    parser = argparse.ArgumentParser(
        description="Compare two data exploration runs side-by-side.",
    )
    parser.add_argument(
        "--dir-a", type=Path,
        default=PROJECT_DIR / "results" / "data" / "data_exploration_phase",
        help="Path to first exploration output directory.",
    )
    parser.add_argument(
        "--dir-b", type=Path,
        default=PROJECT_DIR / "results" / "data" / "data_exploration_phase_merged",
        help="Path to second exploration output directory.",
    )
    parser.add_argument(
        "--label-a", type=str, default=None,
        help="Display label for dataset A (default: directory name).",
    )
    parser.add_argument(
        "--label-b", type=str, default=None,
        help="Display label for dataset B (default: directory name).",
    )
    parser.add_argument(
        "--clinical", type=Path,
        default=DATA_DIR / "Train_clinical.tsv",
        help="Path to clinical labels TSV.",
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="Output tag. Figure saved to compare_explorations_{tag}/.",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("COMPARE EXPLORATIONS")
    log.info("=" * 60)

    # Validate directories.
    for name, dir_path in [("dir-a", args.dir_a), ("dir-b", args.dir_b)]:
        if not dir_path.is_dir():
            log.error("%s is not a directory: %s", name, dir_path)
            sys.exit(1)
        missing = validate_directory(dir_path)
        if missing:
            log.error(
                "%s is missing required files: %s\n"
                "Expected files: %s\n"
                "Run data_exploration_phase.py first to generate them.",
                dir_path, ", ".join(missing), ", ".join(REQUIRED_FILES),
            )
            sys.exit(1)

    # Load data.
    data_a = load_exploration(args.dir_a)
    data_b = load_exploration(args.dir_b)
    clinical_df = pd.read_csv(args.clinical, sep="\t")

    label_a = args.label_a or args.dir_a.name
    label_b = args.label_b or args.dir_b.name

    log.info("A: %s (%d features)", label_a, len(data_a["kw"]))
    log.info("B: %s (%d features)", label_b, len(data_b["kw"]))
    log.info("")

    # Output directory.
    suffix = f"_{args.tag}" if args.tag else ""
    fig_dir = PROJECT_DIR / "results" / "figures" / f"compare_explorations{suffix}"
    fig_dir.mkdir(parents=True, exist_ok=True)

    make_comparison_figure(
        data_a, data_b, label_a, label_b, clinical_df,
        fig_dir / "01_comparison.png",
    )

    log.info("")
    log.info("=" * 60)
    log.info("COMPARISON COMPLETE.")
    log.info("  Figure: %s", fig_dir)
    log.info("  Log:    %s",
             PROJECT_DIR / "results" / "logs" / "compare_explorations.log")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
