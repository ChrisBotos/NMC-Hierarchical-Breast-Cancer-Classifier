"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: pipeline_flowchart.py.
Description:
    Generates a publication-quality flowchart of the full project pipeline,
    from data loading through label-free region merging, nested CV with the
    2x2 grid (feature selection x classifier), and final prediction.

Usage:
    python3 code/pipeline_flowchart.py

Dependencies:
    Python >= 3.10.
    matplotlib.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_box(ax, xy, w, h, text, color, text_color="white", fontsize=9,
             fontweight="bold", alpha=1.0, edgecolor=None, text_lines=None,
             box_style="round,pad=0.02"):
    """Draw a rounded rectangle with centred text.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        xy (tuple): Bottom-left corner (x, y).
        w (float): Width.
        h (float): Height.
        text (str): Primary label.
        color (str): Fill colour.
        text_color (str): Font colour.
        fontsize (int): Font size.
        fontweight (str): Font weight.
        alpha (float): Box opacity.
        edgecolor (str): Edge colour override.
        text_lines (list): Optional extra lines below the main label.
        box_style (str): Matplotlib box style string.

    Returns:
        None.
    """
    ec = edgecolor if edgecolor else color
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=box_style,
        facecolor=color, edgecolor=ec,
        linewidth=1.5, alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    if text_lines:
        line_gap = fontsize * 0.015
        total = len(text_lines)
        top_y = cy + line_gap * total / 2 + line_gap * 0.3
        ax.text(cx, top_y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=3)
        for i, line in enumerate(text_lines):
            ax.text(cx, top_y - (i + 1) * line_gap, line,
                    ha="center", va="center",
                    fontsize=fontsize - 1.5, fontweight="normal",
                    color=text_color, zorder=3, style="italic")
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=text_color,
                zorder=3)


def draw_arrow(ax, start, end, color="#555555", lw=1.5, style="->",
               connectionstyle="arc3,rad=0"):
    """Draw a curved arrow between two points.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        start (tuple): Start (x, y).
        end (tuple): End (x, y).
        color (str): Arrow colour.
        lw (float): Line width.
        style (str): Arrow style.
        connectionstyle (str): Matplotlib connection style string.

    Returns:
        None.
    """
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style, color=color, lw=lw,
        connectionstyle=connectionstyle,
        mutation_scale=14, zorder=1,
    )
    ax.add_patch(arrow)


def phase_label(ax, x, y, text, color):
    """Draw a phase header label.

    Args:
        ax (matplotlib.axes.Axes): Target axes.
        x (float): Centre x.
        y (float): Centre y.
        text (str): Label text.
        color (str): Text colour.

    Returns:
        None.
    """
    ax.text(x, y, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=color, zorder=3)


def build_flowchart():
    """Build and save the full pipeline flowchart.

    Returns:
        Path: Path to the saved figure.
    """
    # --- Colour palette ---
    C_DATA = "#2E86AB"
    C_PHASE0 = "#A23B72"
    C_PREPROC = "#6A994E"
    C_FEAT = "#F18F01"
    C_MODEL = "#C73E1D"
    C_PRED = "#3B1F2B"
    C_OUTPUT = "#44BBA4"
    C_CV_OUTER = "#7B2D8E"
    C_CV_INNER = "#5C6BC0"
    C_ARROW = "#444444"

    fig, ax = plt.subplots(figsize=(11, 19))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 19)
    ax.axis("off")

    MID = 5.5  # Horizontal midpoint.

    # =================================================================
    # TITLE
    # =================================================================
    ax.text(MID, 18.6,
            "CATS Pipeline: Breast Cancer Subtype Classification from aCGH",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="#1a1a1a")

    # =================================================================
    # ROW 1: Data inputs
    # =================================================================
    y = 17.7
    bw, bh = 2.2, 0.55
    draw_box(ax, (0.5, y), bw, bh, "Train_call.tsv", C_DATA,
             text_lines=["100 samples x 2834 regions"])
    draw_box(ax, (3.2, y), bw, bh, "Train_clinical.tsv", C_DATA,
             text_lines=["100 subtype labels"])
    draw_box(ax, (5.9, y), bw, bh, "Validation_call.tsv", C_DATA,
             text_lines=["57 unlabelled samples"])
    draw_box(ax, (8.5, y), 2.0, bh, "Gene Map", C_DATA,
             text_lines=["hg18 coordinates"])

    # =================================================================
    # PHASE 0
    # =================================================================
    y_p0 = 17.0
    phase_label(ax, MID, y_p0, "PHASE 0: DATA EXPLORATION", C_PHASE0)

    y_exp = 16.2
    draw_box(ax, (1.2, y_exp), 8.1, 0.6, "Data Exploration", C_PHASE0,
             fontsize=10, text_lines=[
                 "Class balance (32/36/32)  |  Autocorrelation r=0.92 at <0.1 Mb",
                 "6 Bonferroni-significant regions (chr17 ERBB2)  |  Silhouette=0.09",
             ])

    for x_src in [1.6, 4.3, 7.0, 9.5]:
        draw_arrow(ax, (x_src, y), (MID, y_exp + 0.6), C_ARROW)

    # =================================================================
    # PHASE 1: PREPROCESSING
    # =================================================================
    y_p1 = 15.5
    phase_label(ax, MID, y_p1, "PHASE 1: LABEL-FREE REGION MERGING", C_PREPROC)
    draw_arrow(ax, (MID, y_exp), (MID, y_p1 + 0.15), C_ARROW)

    y_merge = 14.2
    draw_box(ax, (1.0, y_merge), 9.0, 1.05, "Region Merging (CGHregions-style)",
             C_PREPROC, fontsize=10, text_lines=[
                 "Adjacent same-chromosome regions with Pearson r > 0.8 merged",
                 "Consensus value = median CN across constituent regions",
                 "Label-free: uses only CN values, never class labels -> no leakage",
                 "2834 regions  ->  ~273 consensus segments (10x reduction)",
             ])
    draw_arrow(ax, (MID, y_p1 - 0.15), (MID, y_merge + 1.05), C_ARROW)

    # Small note about applying merge map to validation data.
    ax.text(10.2, y_merge + 0.5,
            "Same merge\nmap applied\nto validation\ndata", ha="left",
            va="center", fontsize=6.5, color=C_PREPROC, style="italic")

    # =================================================================
    # PHASE 2: MODEL TRAINING & EVALUATION
    # =================================================================
    y_p2 = 13.4
    phase_label(ax, MID, y_p2,
                "PHASE 2: MODEL TRAINING & EVALUATION", C_MODEL)
    draw_arrow(ax, (MID, y_merge), (MID, y_p2 + 0.15), C_ARROW)

    # --- Outer CV wrapper ---
    outer_x, outer_y, outer_w, outer_h = 0.3, 5.5, 10.4, 7.5
    outer_box = FancyBboxPatch(
        (outer_x, outer_y), outer_w, outer_h,
        boxstyle="round,pad=0.05",
        facecolor="#F3E5F5", edgecolor=C_CV_OUTER,
        linewidth=2.5, linestyle="--", alpha=0.4, zorder=0,
    )
    ax.add_patch(outer_box)
    ax.text(MID, outer_y + outer_h - 0.25,
            "Outer loop: Stratified 5-Fold CV x 10 repeats  (50 splits, estimates AUROC)",
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            color=C_CV_OUTER, zorder=3)
    draw_arrow(ax, (MID, y_p2 - 0.15), (MID, outer_y + outer_h - 0.45),
               C_ARROW)

    # --- Inner CV wrapper ---
    inner_x, inner_y, inner_w, inner_h = 0.7, 8.3, 9.6, 4.2
    inner_box = FancyBboxPatch(
        (inner_x, inner_y), inner_w, inner_h,
        boxstyle="round,pad=0.04",
        facecolor="#E8EAF6", edgecolor=C_CV_INNER,
        linewidth=2.0, linestyle=":", alpha=0.4, zorder=0,
    )
    ax.add_patch(inner_box)
    ax.text(MID, inner_y + inner_h - 0.22,
            "Inner loop: 5-Fold CV on outer training set  (tunes hyperparameters, scores AUROC)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_CV_INNER, zorder=3)

    # --- Feature Selection boxes inside inner loop ---
    fs_y = 10.9
    fs_w, fs_h = 4.3, 1.0
    draw_box(ax, (0.9, fs_y), fs_w, fs_h, "Univariate: K-W top-k", C_FEAT,
             fontsize=9.5, text_lines=[
                 "Rank segments by Kruskal-Wallis p-value",
                 "Select top k segments",
                 "Tune: k in {5,10,15,20,30,50,75,100}",
             ])
    draw_box(ax, (5.8, fs_y), fs_w, fs_h, "Multivariate: Elastic Net", C_FEAT,
             fontsize=9.5, text_lines=[
                 "L1+L2 logistic regression (multinomial)",
                 "Grouping effect for correlated features",
                 "Tune: C x l1_ratio (10 x 5 grid)",
             ])

    ax.text(MID, 11.95, "Feature Selection (on inner training folds only)",
            ha="center", fontsize=8, fontweight="bold", color=C_FEAT,
            zorder=3)

    # --- Classifier boxes inside inner loop ---
    cl_y = 8.7
    cl_w, cl_h = 4.3, 0.85
    draw_box(ax, (0.9, cl_y), cl_w, cl_h, "Simple: NMC", C_MODEL,
             fontsize=9.5, text_lines=[
                 "Nearest Mean Classifier",
                 "Zero hyperparameters, linear boundary",
             ])
    draw_box(ax, (5.8, cl_y), cl_w, cl_h, "Complex: Random Forest", C_MODEL,
             fontsize=9.5, text_lines=[
                 "500 trees, min_samples_leaf=2",
                 "Fixed params (no tuning at n=100)",
             ])

    ax.text(MID, 9.6, "Classifier (fixed parameters)",
            ha="center", fontsize=8, fontweight="bold", color=C_MODEL,
            zorder=3)

    # Arrows from feature selection to classifiers.
    draw_arrow(ax, (3.1, fs_y), (3.1, 9.65), C_ARROW)
    draw_arrow(ax, (7.95, fs_y), (7.95, 9.65), C_ARROW)

    # --- 2x2 cross-connections (dotted, light) ---
    # Left FS -> Right classifier.
    draw_arrow(ax, (5.2, fs_y + 0.1), (5.8, 9.55 + 0.1), "#999999",
               lw=1.0, connectionstyle="arc3,rad=-0.15")
    # Right FS -> Left classifier.
    draw_arrow(ax, (5.8, fs_y + 0.1), (5.2, 9.55 + 0.1), "#999999",
               lw=1.0, connectionstyle="arc3,rad=-0.15")

    # --- 2x2 grid labels in bottom margin of inner loop ---
    grid_label_y = 8.45
    ax.text(MID, grid_label_y,
            "= 4 pipelines:  K-W+NMC  |  K-W+RF  |  ENet+NMC  |  ENet+RF",
            ha="center", va="center", fontsize=7.5, fontweight="bold",
            color="#555555", zorder=3)

    # --- Outer loop evaluation block ---
    y_eval = 6.0
    draw_box(ax, (1.5, y_eval), 8.0, 1.0, "Evaluate on Outer Test Fold",
             "#5E548E", fontsize=10, text_lines=[
                 "Best hyperparams from inner loop -> retrain on full outer training set",
                 "Score AUROC on held-out outer test fold -> one score per pipeline",
                 "50 outer folds -> mean +/- std AUROC per pipeline",
             ])
    draw_arrow(ax, (MID, inner_y), (MID, y_eval + 1.0), C_ARROW)

    # --- 2x2 Results table ---
    y_table = 5.6
    ax.text(MID, y_table, "2x2 AUROC Table  +  Confusion Matrices  +  "
            "Statistical Comparison",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_CV_OUTER, zorder=3, style="italic")

    # =================================================================
    # PHASE 3: FINAL PREDICTION
    # =================================================================
    y_p3 = 4.9
    phase_label(ax, MID, y_p3, "PHASE 3: FINAL PREDICTION", C_PRED)
    draw_arrow(ax, (MID, outer_y), (MID, y_p3 + 0.15), C_ARROW)

    y_best = 3.5
    draw_box(ax, (1.2, y_best), 8.6, 1.1, "Train Final Model", C_PRED,
             text_color="white", fontsize=10, text_lines=[
                 "Select best pipeline from 2x2 AUROC results",
                 "Tune hyperparams (k, or C + l1_ratio) via 5-fold CV on all 100 samples",
                 "Train final model on all 100 samples with best hyperparams",
                 "Predict 57 validation samples",
             ])
    draw_arrow(ax, (MID, y_p3 - 0.15), (MID, y_best + 1.1), C_ARROW)

    # =================================================================
    # Deliverables
    # =================================================================
    y_out = 2.0
    out_w = 2.0
    outputs = [
        ("prediction.txt", "57 predicted labels"),
        ("estimate.txt", "Accuracy estimate"),
        ("model.pkl", "Serialised classifier"),
        ("run_model.py", "Prediction script"),
    ]
    x_positions = [0.7, 3.0, 5.3, 7.9]

    for x, (name, desc) in zip(x_positions, outputs):
        draw_box(ax, (x, y_out), out_w, 0.55, name, C_OUTPUT,
                 fontsize=8.5, text_lines=[desc])

    for x in x_positions:
        draw_arrow(ax, (MID, y_best), (x + out_w / 2, y_out + 0.55),
                   C_ARROW)

    # =================================================================
    # Validation
    # =================================================================
    y_val = 0.8
    draw_box(ax, (2.2, y_val), 6.6, 0.7, "Validation", C_OUTPUT,
             edgecolor="#2E7D32", fontsize=10, text_lines=[
                 "run_model.py reproduces prediction.txt  |  format check passes",
             ])
    draw_arrow(ax, (MID, y_out), (MID, y_val + 0.7), C_ARROW)

    # =================================================================
    # Save
    # =================================================================
    out_path = FIG_DIR / "pipeline_flowchart.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = build_flowchart()
    print(f"Flowchart saved to: {path}")
