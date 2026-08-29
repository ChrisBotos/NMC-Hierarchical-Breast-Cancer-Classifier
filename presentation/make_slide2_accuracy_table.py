"""
Group 9.
Authors:
    Alexandros Michailidis.
    Antonie Wagner.
    Christos Botos.
    Yan Qiao.
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: make_slide2_accuracy_table.py.
Description:
    Generates a two-panel figure for presentation slide 2: a comparative
    accuracy table (left) and a BA2 strip plot (right) showing the
    distribution of Stage 2 balanced accuracy across 200 repeats for
    each pipeline.

Usage:
    python3 presentation/make_slide2_accuracy_table.py

Dependencies:
    Python >= 3.10.
    pandas, numpy, matplotlib.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Colour palette."""
C = {
    "stage1": "#4DBBD5",
    "stage2": "#F39B7F",
    "highlight": "#FFD700",
    "highlight_bg": "#FFF8DC",
    "border": "#444444",
    "text": "#1A1A1A",
    "header_bg": "#3C3C3C",
    "header_text": "#FFFFFF",
    "row_even": "#F8F8F8",
    "row_odd": "#FFFFFF",
    "best_row": "#FFF3CD",
    "HER2": "#E64B35",
    "HR": "#3C5488",
    "TN": "#00A087",
}

"""Pipeline display names mapping."""
DISPLAY_NAMES = {
    "en_nmc_pens": "EN + NMC (plat.)",
    "nmc_pens_ensemble": "NMC Pens Ens.",
    "kw_nmc_pens": "KW + NMC (plat.)",
    "standalone_en_pens": "Standalone EN (plat.)",
    "nmc_ensemble": "NMC Ensemble",
    "standalone_en": "Standalone EN",
    "kw_nmc": "KW + NMC",
    "en_nmc": "EN + NMC",
    "kw_rf": "KW + RF",
    "en_rf": "EN + RF",
}


"""Pipelines without a meaningful single feature count."""
ENSEMBLE_PIPELINES = {
    "nmc_pens_ensemble",
    "nmc_ensemble",
    "standalone_en_pens",
    "standalone_en",
}

SUBMITTED_PIPELINE = "en_nmc_pens"

"""Table layout constants in figure fraction coordinates."""
TABLE_LEFT = 0.01
TABLE_RIGHT = 0.60
STRIP_LEFT = 0.63
STRIP_RIGHT = 0.98
TABLE_TOP = 0.93
ROW_HEIGHT = 0.072
HEADER_HEIGHT = 0.085

"""Column x-positions as fractions of the table width (0 to 1)."""
COL_POSITIONS = {
    "rank": (0.00, 0.035),
    "pipeline": (0.035, 0.250),
    "ba2": (0.250, 0.470),
    "comb_ba": (0.470, 0.690),
    "auroc": (0.690, 0.790),
    "acc": (0.790, 0.890),
    "feat": (0.890, 1.000),
}


def load_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three source data files.

    Args:
        project_root (Path): Project root directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: BA2 summary,
            combined BA summary, and all fold results.
    """
    base = (
        project_root
        / "results"
        / "2026-04-25_final_hierarchical"
        / "nested_cv_analysis"
        / "data"
    )
    ba2_summary = pd.read_csv(base / "summary_statistics.csv")
    combined_summary = pd.read_csv(base / "summary_statistics_combined_ba.csv")
    all_folds = pd.read_csv(
        base / "all_fold_results.csv",
        usecols=[
            "pipeline",
            "repeat",
            "outer_fold",
            "stage2_bal_acc",
            "balanced_accuracy",
            "y_true",
            "y_pred",
        ],
    )

    # Compute per-fold plain accuracy from y_true and y_pred.
    def _fold_acc(row):
        yt = row["y_true"].split(",")
        yp = row["y_pred"].split(",")
        return sum(t == p for t, p in zip(yt, yp)) / len(yt)

    all_folds["accuracy"] = all_folds.apply(_fold_acc, axis=1)

    # Compute mean accuracy per pipeline and merge into combined summary.
    acc_by_pipe = (
        all_folds.groupby("pipeline")["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "mean_accuracy"})
    )
    combined_summary = combined_summary.merge(acc_by_pipe, on="pipeline", how="left")

    return ba2_summary, combined_summary, all_folds


def compute_repeat_means(all_folds: pd.DataFrame) -> pd.DataFrame:
    """Compute mean BA2 per (pipeline, repeat) by averaging across outer folds.

    Args:
        all_folds (pd.DataFrame): Per-fold results.

    Returns:
        pd.DataFrame: One row per (pipeline, repeat), with mean_ba2 column.
    """
    repeat_means = (
        all_folds.groupby(["pipeline", "repeat"])["stage2_bal_acc"]
        .mean()
        .reset_index()
        .rename(columns={"stage2_bal_acc": "mean_ba2"})
    )
    return repeat_means


def build_table_data(
    ba2_summary: pd.DataFrame,
    combined_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge BA2 and combined BA summaries into a single table dataframe.

    Args:
        ba2_summary (pd.DataFrame): Stage 2 BA summary statistics.
        combined_summary (pd.DataFrame): Combined 3-class BA summary.

    Returns:
        pd.DataFrame: Merged table data sorted by BA2 descending.
    """
    comb_cols = ["pipeline", "mean_bal_acc", "std_bal_acc"]
    if "mean_accuracy" in combined_summary.columns:
        comb_cols.append("mean_accuracy")
    merged = ba2_summary.merge(
        combined_summary[comb_cols],
        on="pipeline",
        suffixes=("_ba2", "_comb"),
    )
    merged = merged.sort_values("mean_bal_acc_ba2", ascending=False).reset_index(
        drop=True
    )
    return merged


def _table_x_to_fig(frac: float) -> float:
    """Convert table-relative x fraction to figure fraction.

    Args:
        frac (float): Fraction within the table width (0 to 1).

    Returns:
        float: Figure-level x coordinate.
    """
    return TABLE_LEFT + frac * (TABLE_RIGHT - TABLE_LEFT)


def _row_y_center(row_idx: int) -> float:
    """Get the vertical centre of a data row in figure fraction coordinates.

    Row 0 is the first data row (directly below the header).

    Args:
        row_idx (int): Zero-based row index.

    Returns:
        float: Figure-level y coordinate of the row centre.
    """
    return TABLE_TOP - HEADER_HEIGHT - row_idx * ROW_HEIGHT - ROW_HEIGHT / 2


def draw_table(fig: plt.Figure, table_data: pd.DataFrame) -> None:
    """Draw the formatted accuracy comparison table directly on the figure.

    Places rectangles and text using figure-level coordinates for precise
    control over cell sizes and alignment.

    Args:
        fig (plt.Figure): The figure to draw on.
        table_data (pd.DataFrame): Merged summary data, sorted by BA2.
    """
    n_rows = len(table_data)

    # Column definitions: (key, header_label).
    col_defs = [
        ("rank", "#"),
        ("pipeline", "Pipeline"),
        ("ba2", "BA2\n(mean +/- std)"),
        ("comb_ba", "Combined BA\n(mean +/- std)"),
        ("auroc", "AUROC"),
        ("acc", "Acc."),
        ("feat", "Feat."),
    ]

    # Draw header cells.
    header_bot = TABLE_TOP - HEADER_HEIGHT
    for key, label in col_defs:
        x_start_frac, x_end_frac = COL_POSITIONS[key]
        x0 = _table_x_to_fig(x_start_frac)
        x1 = _table_x_to_fig(x_end_frac)
        w = x1 - x0
        h = HEADER_HEIGHT

        rect = mpatches.FancyBboxPatch(
            (x0, header_bot),
            w,
            h,
            boxstyle="square,pad=0",
            facecolor=C["header_bg"],
            edgecolor=C["border"],
            linewidth=0.6,
            transform=fig.transFigure,
            clip_on=False,
        )
        fig.patches.append(rect)

        fig.text(
            x0 + w / 2,
            header_bot + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=C["header_text"],
            fontfamily="sans-serif",
            linespacing=1.2,
        )

    # Draw data rows.
    for i, (_, row) in enumerate(table_data.iterrows()):
        row_top = TABLE_TOP - HEADER_HEIGHT - i * ROW_HEIGHT
        row_bot = row_top - ROW_HEIGHT
        y_center = row_top - ROW_HEIGHT / 2
        pid = row["pipeline"]
        is_submitted = pid == SUBMITTED_PIPELINE

        # Row background colour.
        if is_submitted:
            bg = C["best_row"]
        elif i % 2 == 0:
            bg = C["row_even"]
        else:
            bg = C["row_odd"]

        # Format cell values.
        rank_str = str(i + 1)
        display_name = DISPLAY_NAMES.get(pid, pid)
        if is_submitted:
            display_name += "  *"
        ba2_str = f"{row['mean_bal_acc_ba2']:.3f} +/- {row['std_bal_acc_ba2']:.3f}"
        comb_str = f"{row['mean_bal_acc_comb']:.3f} +/- {row['std_bal_acc_comb']:.3f}"
        auroc_str = f"{row['mean_auroc']:.3f}"
        acc_str = (
            f"{row['mean_accuracy']:.3f}" if pd.notna(row.get("mean_accuracy")) else "-"
        )
        if pid in ENSEMBLE_PIPELINES:
            feat_str = "-"
        else:
            feat_str = f"{row['mean_n_features']:.1f}"

        cell_map = {
            "rank": rank_str,
            "pipeline": display_name,
            "ba2": ba2_str,
            "comb_ba": comb_str,
            "auroc": auroc_str,
            "acc": acc_str,
            "feat": feat_str,
        }

        for key, _ in col_defs:
            x_start_frac, x_end_frac = COL_POSITIONS[key]
            x0 = _table_x_to_fig(x_start_frac)
            x1 = _table_x_to_fig(x_end_frac)
            w = x1 - x0

            # Draw cell background.
            rect = mpatches.FancyBboxPatch(
                (x0, row_bot),
                w,
                ROW_HEIGHT,
                boxstyle="square,pad=0",
                facecolor=bg,
                edgecolor=C["border"],
                linewidth=0.3,
                transform=fig.transFigure,
                clip_on=False,
            )
            fig.patches.append(rect)

            # Text properties.
            if key == "pipeline":
                text_x = x0 + 0.008
                ha = "left"
            else:
                text_x = x0 + w / 2
                ha = "center"

            if key in ("ba2", "comb_ba", "auroc", "acc", "feat"):
                fontfam = "monospace"
                fontsize = 10.5
            elif key == "pipeline":
                fontfam = "sans-serif"
                fontsize = 11
            else:
                fontfam = "sans-serif"
                fontsize = 12

            fontweight = "bold" if is_submitted else "normal"

            fig.text(
                text_x,
                y_center,
                cell_map[key],
                ha=ha,
                va="center",
                fontsize=fontsize,
                fontweight=fontweight,
                color=C["text"],
                fontfamily=fontfam,
            )

        # Gold left-edge accent for submitted row.
        if is_submitted:
            accent = mpatches.FancyBboxPatch(
                (TABLE_LEFT, row_bot),
                0.004,
                ROW_HEIGHT,
                boxstyle="square,pad=0",
                facecolor=C["highlight"],
                edgecolor="none",
                transform=fig.transFigure,
                clip_on=False,
            )
            fig.patches.append(accent)

    # Footnote below the table.
    footnote_y = TABLE_TOP - HEADER_HEIGHT - n_rows * ROW_HEIGHT - 0.02
    fig.text(
        TABLE_LEFT + 0.005,
        footnote_y,
        "* Submitted method",
        ha="left",
        va="top",
        fontsize=10,
        fontstyle="italic",
        color="#666666",
        fontfamily="sans-serif",
    )


def draw_strip_plot(
    fig: plt.Figure,
    repeat_means: pd.DataFrame,
    pipeline_order: list[str],
    ba2_summary: pd.DataFrame,
    n_rows: int,
) -> None:
    """Draw horizontal box/strip plots aligned with the table rows.

    The axes position is computed from the table row layout so that each
    box plot row lines up exactly with its corresponding table row.

    Args:
        fig (plt.Figure): The figure.
        repeat_means (pd.DataFrame): Mean BA2 per (pipeline, repeat).
        pipeline_order (list[str]): Pipeline IDs in display order (top to bottom).
        ba2_summary (pd.DataFrame): Summary statistics for mean markers.
        n_rows (int): Number of pipelines.
    """
    # Compute axes position to match table rows.
    first_row_center = _row_y_center(0)
    last_row_center = _row_y_center(n_rows - 1)
    margin = ROW_HEIGHT * 0.6
    ax_bottom = last_row_center - margin
    ax_top = first_row_center + margin
    ax_height = ax_top - ax_bottom

    ax = fig.add_axes([STRIP_LEFT, ax_bottom, STRIP_RIGHT - STRIP_LEFT, ax_height])

    y_positions = list(range(n_rows))

    # Prepare data arrays per pipeline.
    box_data = []
    for pid in pipeline_order:
        vals = repeat_means.loc[repeat_means["pipeline"] == pid, "mean_ba2"].values
        box_data.append(vals)

    # Draw horizontal box plots.
    bp = ax.boxplot(
        box_data,
        positions=y_positions,
        vert=False,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        whiskerprops=dict(color="#888888", linewidth=0.8),
        capprops=dict(color="#888888", linewidth=0.8),
        medianprops=dict(color=C["text"], linewidth=1.2),
    )

    # Colour the boxes.
    for patch, pid in zip(bp["boxes"], pipeline_order):
        if pid == SUBMITTED_PIPELINE:
            patch.set_facecolor(C["highlight"])
            patch.set_edgecolor(C["border"])
            patch.set_linewidth(1.2)
            patch.set_alpha(0.85)
        else:
            patch.set_facecolor(C["stage2"])
            patch.set_edgecolor(C["border"])
            patch.set_linewidth(0.6)
            patch.set_alpha(0.5)

    # Overlay jittered dots.
    rng = np.random.default_rng(42)
    for i, (pid, vals) in enumerate(zip(pipeline_order, box_data)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        if pid == SUBMITTED_PIPELINE:
            dot_color = "#DAA520"
            dot_alpha = 0.35
            dot_size = 5
        else:
            dot_color = C["stage2"]
            dot_alpha = 0.15
            dot_size = 3
        ax.scatter(
            vals,
            i + jitter,
            s=dot_size,
            color=dot_color,
            alpha=dot_alpha,
            edgecolors="none",
            zorder=2,
        )

    # Plot mean markers.
    means_lookup = ba2_summary.set_index("pipeline")["mean_bal_acc"]
    for i, pid in enumerate(pipeline_order):
        mean_val = means_lookup.loc[pid]
        marker_color = C["text"] if pid != SUBMITTED_PIPELINE else "#B8860B"
        ax.scatter(
            mean_val,
            i,
            s=45,
            color=marker_color,
            marker="D",
            edgecolors="white",
            linewidths=0.5,
            zorder=5,
        )

    # Best mean vertical line.
    best_mean = means_lookup.max()
    ax.axvline(
        best_mean,
        color=C["highlight"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    # Axis formatting.
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""] * n_rows)
    ax.set_xlabel(
        "BA2 (Stage 2 Balanced Accuracy)", fontsize=11, fontfamily="sans-serif"
    )
    ax.invert_yaxis()
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_facecolor("white")

    # Add a panel label.
    ax.set_title(
        "BA2 Distribution (200 repeats)",
        fontsize=11.5,
        fontweight="bold",
        fontfamily="sans-serif",
        pad=8,
    )

    # Legend for mean marker.
    diamond_handle = plt.Line2D(
        [0],
        [0],
        marker="D",
        color="white",
        markerfacecolor=C["text"],
        markeredgecolor="white",
        markersize=6,
        label="Mean",
        linewidth=0,
    )
    dashed_handle = plt.Line2D(
        [0],
        [0],
        color=C["highlight"],
        linestyle="--",
        linewidth=1.2,
        label="Best mean",
    )
    ax.legend(
        handles=[diamond_handle, dashed_handle],
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#CCCCCC",
    )


def main() -> None:
    """Generate the slide 2 accuracy table figure."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Load data.
    ba2_summary, combined_summary, all_folds = load_data(project_root)

    # Compute per-repeat mean BA2.
    repeat_means = compute_repeat_means(all_folds)

    # Build merged table data.
    table_data = build_table_data(ba2_summary, combined_summary)
    pipeline_order = table_data["pipeline"].tolist()
    n_rows = len(pipeline_order)

    # Create figure.
    fig = plt.figure(figsize=(13.33, 7.5), dpi=300)
    fig.patch.set_facecolor("white")

    # Draw both panels.
    draw_table(fig, table_data)
    draw_strip_plot(fig, repeat_means, pipeline_order, ba2_summary, n_rows)

    # Bottom-level footnote spanning full width.
    fig.text(
        0.01,
        0.015,
        "BA2 = Stage 2 balanced accuracy (HR+ vs TN). "
        "Combined BA includes perfect Stage 1 (HER2+ vs rest). "
        "200 repeated stratified 5-fold CV.",
        fontsize=9,
        fontstyle="italic",
        color="#777777",
        fontfamily="sans-serif",
        va="bottom",
    )

    # Save.
    output_path = script_dir / "slide_2_accuracy_table.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
