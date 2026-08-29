"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: plot_decision_boundary.py.
Description:
    Visualise per-sample classification difficulty from the 200-repeat
    nested CV results. Shows Stage 2 (HR+ vs Triple Negative) samples
    in unsupervised PCA space, coloured by cross-validated misclassification
    rate, comparing EN+NMC vs EN+NMC (plateau ensemble).

Usage:
    python3 code/plot_decision_boundary.py --name final_hierarchical

Dependencies:
    Python >= 3.10.
    scikit-learn, pandas, numpy, matplotlib.
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Allow imports from the code directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.cv_io import load_cv_data
from utils.paths import _find_or_create_run_dir

# Subtype colours matching the project palette.
CLASS_COLORS = {"HR+": "#4DBBD5", "Triple Neg": "#00A087"}
CLASS_MARKERS = {"HR+": "o", "Triple Neg": "s"}


def load_sample_info(run_dir):
    """Load training data, labels, and sample names.

    Args:
        run_dir (Path): Run directory containing preprocessing/data/.

    Returns:
        tuple: (X, y_labels, sample_names) where X is the feature matrix,
            y_labels is an array of string labels, and sample_names is a
            list of sample identifiers.
    """
    merged_path = run_dir / "preprocessing" / "data" / "train_merged.tsv"
    clinical_path = (
        Path(__file__).resolve().parent.parent / "data" / "Train_clinical.tsv"
    )

    X, y, le, feature_names, sample_names = load_cv_data(
        merged_path,
        clinical_path,
    )
    y_labels = le.inverse_transform(y)
    return X, y_labels, sample_names


def compute_misclassification_rates(data_dir, pipeline_name, n_samples):
    """Compute per-sample misclassification rate from CV fold results.

    For each sample, counts how many times it appeared in a test fold
    and how many times it was misclassified. Returns the rate for each
    of the n_samples.

    Args:
        data_dir (Path): Directory containing fold_results CSVs.
        pipeline_name (str): Pipeline name (e.g. 'en_nmc').
        n_samples (int): Total number of samples (100).

    Returns:
        tuple: (misclass_rate, n_tested) where misclass_rate[i] is the
            fraction of test appearances where sample i was misclassified,
            and n_tested[i] is how many times sample i was tested.
    """
    pattern = str(data_dir / f"fold_results_{pipeline_name}_r*.csv")
    files = sorted(glob.glob(pattern))

    n_correct = np.zeros(n_samples)
    n_tested = np.zeros(n_samples)

    for f in files:
        df = pd.read_csv(f)
        for _, row in df.iterrows():
            indices = [int(x) for x in row["test_indices"].split(",")]
            y_true = row["y_true"].split(",")
            y_pred = row["y_pred"].split(",")

            for idx, true, pred in zip(indices, y_true, y_pred):
                n_tested[idx] += 1
                if true == pred:
                    n_correct[idx] += 1

    # Avoid division by zero for samples never tested (should not happen).
    with np.errstate(invalid="ignore"):
        misclass_rate = 1.0 - (n_correct / n_tested)
    misclass_rate = np.nan_to_num(misclass_rate, nan=0.0)

    return misclass_rate, n_tested


def plot_misclassification(X_pca, y_labels, misclass_en, misclass_pens, pca, fig_path):
    """Create the side-by-side per-sample misclassification figure.

    Args:
        X_pca (np.ndarray): PCA-projected coordinates for all samples (n, 2).
        y_labels (np.ndarray): String labels for all samples.
        misclass_en (np.ndarray): Per-sample misclassification rate for en_nmc.
        misclass_pens (np.ndarray): Per-sample misclass rate for en_nmc_pens.
        pca (PCA): Fitted PCA for variance labels.
        fig_path (Path): Output path for the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    var_explained = pca.explained_variance_ratio_ * 100

    # Colour map: green (correct) to red (misclassified).
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.Normalize(vmin=0, vmax=0.6)

    # Stage 2 mask (non-HER2+).
    s2_mask = y_labels != "HER2+"
    her2_mask = y_labels == "HER2+"

    panels = [
        (axes[0], misclass_en, "EN + NMC"),
        (axes[1], misclass_pens, "EN + NMC (plateau ensemble)"),
    ]

    for ax, misclass, label in panels:
        # Plot HER2+ samples as small grey dots (always correct).
        ax.scatter(
            X_pca[her2_mask, 0],
            X_pca[her2_mask, 1],
            c="#CCCCCC",
            marker="D",
            s=25,
            edgecolors="#999999",
            linewidths=0.4,
            label="HER2+ (always correct)",
            zorder=3,
        )

        # Plot Stage 2 samples coloured by misclassification rate.
        for cls_name, marker in CLASS_MARKERS.items():
            cls_mask = y_labels == cls_name
            ax.scatter(
                X_pca[cls_mask, 0],
                X_pca[cls_mask, 1],
                c=misclass[cls_mask],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=70,
                edgecolors="k",
                linewidths=0.6,
                label=cls_name,
                zorder=5,
            )

        ax.set_xlabel(
            f"PC1 ({var_explained[0]:.1f}% variance)",
            fontsize=10,
        )
        ax.set_xlim(X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1)
        ax.set_ylim(X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1)

        # Annotate mean Stage 2 misclassification rate.
        mean_misclass = misclass[s2_mask].mean()
        ax.text(
            0.97,
            0.97,
            f"Mean misclass.: {mean_misclass:.1%}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="#CCCCCC",
                alpha=0.9,
            ),
        )

        # Panel label below the plot.
        ax.text(
            0.5,
            -0.16,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    axes[0].set_ylabel(
        f"PC2 ({var_explained[1]:.1f}% variance)",
        fontsize=10,
    )
    axes[0].legend(
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        markerscale=0.8,
    )

    # Shared colour bar.
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.8,
        pad=0.02,
    )
    cbar.set_label(
        "CV misclassification rate (200 repeats x 5 folds)",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.05, 0.92, 1])
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")


def main():
    """Generate the per-sample misclassification comparison figure."""
    parser = argparse.ArgumentParser(
        description="Plot per-sample CV misclassification for EN+NMC variants.",
    )
    parser.add_argument(
        "--name",
        default="final_hierarchical",
        help="Run name (default: final_hierarchical).",
    )
    args = parser.parse_args()

    run_dir = _find_or_create_run_dir(args.name)
    fig_dir = run_dir / "hierarchical_nested_cv" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_dir = run_dir / "hierarchical_nested_cv" / "data"

    # Load training data for PCA projection.
    X, y_labels, sample_names = load_sample_info(run_dir)
    n_samples = len(y_labels)
    s2_mask = y_labels != "HER2+"
    print(
        f"Total samples: {n_samples} "
        f"(HER2+: {(~s2_mask).sum()}, Stage 2: {s2_mask.sum()})"
    )

    # Unsupervised PCA on all 100 samples (no label leakage).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(
        f"PCA variance explained: "
        f"{pca.explained_variance_ratio_.sum()*100:.1f}% "
        f"({pca.explained_variance_ratio_[0]*100:.1f}% + "
        f"{pca.explained_variance_ratio_[1]*100:.1f}%)"
    )

    # Compute per-sample misclassification rates from CV fold results.
    print("Computing misclassification rates for en_nmc...")
    misclass_en, n_tested_en = compute_misclassification_rates(
        data_dir,
        "en_nmc",
        n_samples,
    )
    print(
        f"  Stage 2 mean misclass: {misclass_en[s2_mask].mean():.3f}, "
        f"  each sample tested ~{n_tested_en[s2_mask].mean():.0f} times"
    )

    print("Computing misclassification rates for en_nmc_pens...")
    misclass_pens, n_tested_pens = compute_misclassification_rates(
        data_dir,
        "en_nmc_pens",
        n_samples,
    )
    print(
        f"  Stage 2 mean misclass: {misclass_pens[s2_mask].mean():.3f}, "
        f"  each sample tested ~{n_tested_pens[s2_mask].mean():.0f} times"
    )

    # Plot.
    fig_path = fig_dir / "misclassification_en_nmc_vs_pens.png"
    plot_misclassification(
        X_pca,
        y_labels,
        misclass_en,
        misclass_pens,
        pca,
        fig_path,
    )


if __name__ == "__main__":
    main()
