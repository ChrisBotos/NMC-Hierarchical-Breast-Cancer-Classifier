"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: make_slide1_cv_scheme.py.
Description:
    Generates a schematic diagram of the nested cross-validation and
    hierarchical classifier architecture for presentation slide 1.
    Pure matplotlib patches, arrows, and text - no data plots.

Usage:
    python3 presentation/make_slide1_cv_scheme.py

Dependencies:
    Python >= 3.10.
    matplotlib.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Shared colour palette (consistent across all 4 slides).
C = {
    'stage1': '#4DBBD5',
    'stage1_dark': '#2D7A8E',
    'stage2': '#F39B7F',
    'stage2_dark': '#C06040',
    'preproc': '#91D1C2',
    'preproc_dark': '#5DA08E',
    'cv_outer': '#E8E8E8',
    'cv_inner': '#D0D0D0',
    'train': '#B8D4E3',
    'test': '#F5B7B1',
    'highlight': '#FFD700',
    'border': '#444444',
    'text': '#1A1A1A',
    'arrow': '#1A1A1A',
    'HER2': '#E64B35',
    'HR': '#3C5488',
    'TN': '#00A087',
}

ARROW_KW = dict(
    arrowstyle='->', mutation_scale=15, color=C['arrow'], lw=1.5,
    connectionstyle='arc3,rad=0',
)

ARROW_KW_THIN = dict(
    arrowstyle='->', mutation_scale=12, color=C['arrow'], lw=1.2,
    connectionstyle='arc3,rad=0',
)


def _box(ax, x, y, w, h, fc, ec=None, lw=1.2, zorder=2, alpha=1.0,
         pad=0.008):
    """Draw a rounded rectangle and return it."""
    ec = ec or C['border']
    b = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
        alpha=alpha,
    )
    ax.add_patch(b)
    return b


def _arrow(ax, xy_from, xy_to, **kw):
    """Draw a fancy arrow between two points."""
    merged = {**ARROW_KW, **kw}
    a = FancyArrowPatch(xy_from, xy_to, **merged)
    ax.add_patch(a)
    return a


def _centered_text(ax, x, y, txt, **kw):
    """Place text centred at (x, y)."""
    defaults = dict(ha='center', va='center', color=C['text'], zorder=5)
    defaults.update(kw)
    return ax.text(x, y, txt, **defaults)


# ── Column 1: Data and Preprocessing ────────────────────────────────────────

def _draw_column1(ax):
    """Draw the data and preprocessing column."""
    col_l = 0.02
    col_r = 0.24
    col_cx = (col_l + col_r) / 2
    bw = col_r - col_l  # Box width.
    bh = 0.13

    # Bracket label at the top.
    _centered_text(ax, col_cx + 0.02, 0.96, "DATA PREPARATION",
                   fontsize=14, fontweight='bold', color=C['text'])
    _centered_text(ax, col_cx + 0.02, 0.92, "(outside CV, label-free)",
                   fontsize=10, fontstyle='italic', color=C['text'])

    # Box 1 - Raw aCGH data.
    y1 = 0.68
    _box(ax, col_l, y1, bw, bh, fc=C['preproc'], ec=C['preproc_dark'])
    _centered_text(ax, col_cx, y1 + bh - 0.025, "Raw aCGH Data",
                   fontsize=13, fontweight='bold')
    _centered_text(ax, col_cx, y1 + bh / 2 - 0.01,
                   "100 samples x 2,834 regions", fontsize=10)
    _centered_text(ax, col_cx, y1 + 0.02,
                   "Discrete CN: {-1, 0, +1, +2}", fontsize=10)

    # Arrow 1->2.
    _arrow(ax, (col_cx, y1), (col_cx, y1 - 0.055))

    # Box 2 - Region merging.
    y2 = 0.48
    _box(ax, col_l, y2, bw, bh, fc=C['preproc'], ec=C['preproc_dark'])
    _centered_text(ax, col_cx, y2 + bh - 0.025, "Region Merging",
                   fontsize=13, fontweight='bold')
    _centered_text(ax, col_cx, y2 + bh / 2 - 0.01,
                   "Pearson r > 0.8 threshold", fontsize=10)
    _centered_text(ax, col_cx, y2 + 0.02,
                   "Label-free (no target leakage)", fontsize=10)

    # Arrow 2->3.
    _arrow(ax, (col_cx, y2), (col_cx, y2 - 0.055))

    # Box 3 - Merged data.
    y3 = 0.28
    _box(ax, col_l, y3, bw, bh, fc=C['preproc'], ec=C['preproc_dark'])
    _centered_text(ax, col_cx, y3 + bh - 0.025, "Merged Data",
                   fontsize=13, fontweight='bold')
    _centered_text(ax, col_cx, y3 + bh / 2 - 0.01,
                   "100 samples x 273 segments", fontsize=10)
    _centered_text(ax, col_cx, y3 + 0.02,
                   "Input to nested CV", fontsize=10, fontstyle='italic')

    # Arrow from merged data to column 2.
    _arrow(ax, (col_r + 0.005, y3 + bh / 2), (0.27, y3 + bh / 2),
           lw=2.0, color=C['text'])


# ── Column 2: Nested CV Architecture ────────────────────────────────────────

def _draw_column2(ax):
    """Draw the nested CV architecture column."""
    col_l = 0.28
    col_r = 0.63
    col_cx = (col_l + col_r) / 2
    bw = col_r - col_l

    # Outer CV box.
    outer_y = 0.04
    outer_h = 0.92
    _box(ax, col_l, outer_y, bw, outer_h, fc=C['cv_outer'],
         ec=C['border'], lw=1.8, zorder=1, pad=0.012)

    # Header.
    _centered_text(ax, col_cx, 0.935,
                   "REPEATED STRATIFIED 5-FOLD CV",
                   fontsize=14, fontweight='bold')
    _centered_text(ax, col_cx, 0.895,
                   "200 repeats (seeds 1001 - 1200)",
                   fontsize=11)

    # Fold bars - show one representative repeat.
    _centered_text(ax, col_cx, 0.860,
                   "One repeat (5 outer folds):", fontsize=10,
                   fontstyle='italic')

    bar_y_start = 0.79
    bar_h = 0.032
    bar_gap = 0.005
    bar_l = col_l + 0.02
    bar_total_w = bw - 0.04
    fold_w = bar_total_w  # Full width for each bar.

    for i in range(5):
        by = bar_y_start - i * (bar_h + bar_gap)
        test_fold_idx = i  # Which segment is the test fold.

        # Draw 5 sub-segments within each bar.
        seg_w = fold_w / 5
        for j in range(5):
            sx = bar_l + j * seg_w
            if j == test_fold_idx:
                fc = C['test']
                ec_seg = '#D08080'
            else:
                fc = C['train']
                ec_seg = '#8BAEC0'
            rect = FancyBboxPatch(
                (sx, by), seg_w - 0.002, bar_h,
                boxstyle="round,pad=0.003",
                facecolor=fc, edgecolor=ec_seg, linewidth=0.8, zorder=3,
            )
            ax.add_patch(rect)

        # Label only the first bar.
        if i == 0:
            # Identify segments.
            for j in range(5):
                sx = bar_l + j * seg_w + seg_w / 2
                if j == test_fold_idx:
                    _centered_text(ax, sx, by + bar_h + 0.015,
                                   "Test", fontsize=9, fontweight='bold',
                                   color=C['text'])
                elif j == 1:
                    _centered_text(ax, sx, by + bar_h + 0.015,
                                   "Train", fontsize=9, fontweight='bold',
                                   color=C['text'])

        # Fold label on right.
        _centered_text(ax, col_r - 0.01, by + bar_h / 2,
                       f"F{i+1}", fontsize=9, ha='left',
                       color=C['text'])

    # Divider line.
    div_y = 0.575
    ax.plot([col_l + 0.015, col_r - 0.015], [div_y, div_y],
            color='#AAAAAA', lw=0.8, ls='--', zorder=3)

    # Expanded view: what happens in one fold.
    _centered_text(ax, col_cx, 0.55, "Within each outer fold:",
                   fontsize=11, fontweight='bold')

    # Training box.
    train_bx = col_l + 0.02
    train_by = 0.40
    train_bw = bw / 2 - 0.04
    train_bh = 0.12
    _box(ax, train_bx, train_by, train_bw, train_bh,
         fc=C['train'], ec='#8BAEC0', lw=1.0)
    train_cx = train_bx + train_bw / 2
    _centered_text(ax, train_cx, train_by + train_bh - 0.025,
                   "80% Train", fontsize=11, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, train_cx, train_by + 0.04,
                   "Fit hierarchical\nclassifier", fontsize=9,
                   color=C['text'])

    # Test box.
    test_bx = col_l + bw / 2 + 0.02
    test_by = 0.40
    test_bw = bw / 2 - 0.04
    test_bh = 0.12
    _box(ax, test_bx, test_by, test_bw, test_bh,
         fc=C['test'], ec='#D08080', lw=1.0)
    test_cx = test_bx + test_bw / 2
    _centered_text(ax, test_cx, test_by + test_bh - 0.025,
                   "20% Test", fontsize=11, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, test_cx, test_by + 0.04,
                   "Evaluate:\nBA (3-class)", fontsize=9,
                   color=C['text'])

    # Inner CV box - nested inside training.
    inner_x = col_l + 0.025
    inner_y = 0.21
    inner_w = bw / 2 - 0.05
    inner_h = 0.13
    _box(ax, inner_x, inner_y, inner_w, inner_h,
         fc=C['cv_inner'], ec='#999999', lw=1.0, zorder=2)
    inner_cx = inner_x + inner_w / 2
    _centered_text(ax, inner_cx, inner_y + inner_h - 0.02,
                   "5-Fold Inner CV", fontsize=10, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, inner_cx, inner_y + inner_h / 2 - 0.01,
                   "Tunes Stage 2", fontsize=9, color=C['text'])
    _centered_text(ax, inner_cx, inner_y + 0.02,
                   "Scoring: balanced acc.", fontsize=9,
                   color=C['text'], fontstyle='italic')

    # Arrow from train to inner CV.
    _arrow(ax, (train_cx, train_by), (inner_cx, inner_y + inner_h + 0.005),
           **ARROW_KW_THIN)

    # Metric aggregation info at bottom right of fold view.
    metric_cx = test_cx
    _centered_text(ax, metric_cx, 0.28,
                   "Primary metric:", fontsize=9, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, metric_cx, 0.25,
                   "Stage 2 BA (BA2)", fontsize=9,
                   color=C['text'])

    # Bottom summary.
    _centered_text(ax, col_cx, 0.13,
                   "200 repeats x 5 folds = 1,000 estimates",
                   fontsize=12, fontweight='bold')
    _centered_text(ax, col_cx, 0.085,
                   "Report: mean +/- std", fontsize=11,
                   fontstyle='italic')


# ── Column 3: Hierarchical Classifier Architecture ──────────────────────────

def _draw_column3(ax):
    """Draw the hierarchical classifier architecture."""
    col_l = 0.66
    col_r = 0.98
    col_cx = (col_l + col_r) / 2
    bw = col_r - col_l

    # Header.
    _centered_text(ax, col_cx, 0.97, "HIERARCHICAL CLASSIFIER",
                   fontsize=14, fontweight='bold', color=C['text'])

    # Input box.
    inp_w = bw * 0.7
    inp_h = 0.05
    inp_x = col_cx - inp_w / 2
    inp_y = 0.88
    _box(ax, inp_x, inp_y, inp_w, inp_h, fc='#F0F0F0', ec=C['border'])
    _centered_text(ax, col_cx, inp_y + inp_h / 2, "Test samples",
                   fontsize=11, fontweight='bold')

    # Arrow input -> Stage 1.
    _arrow(ax, (col_cx, inp_y), (col_cx, inp_y - 0.015))

    # Stage 1 box.
    s1_w = bw - 0.01
    s1_h = 0.16
    s1_x = col_l + 0.005
    s1_y = 0.70
    _box(ax, s1_x, s1_y, s1_w, s1_h, fc=C['stage1'], ec=C['stage1_dark'],
         lw=1.5)
    s1_cx = s1_x + s1_w / 2
    _centered_text(ax, s1_cx, s1_y + s1_h - 0.022,
                   "STAGE 1", fontsize=13, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, s1_cx, s1_y + s1_h - 0.05,
                   "HER2+ vs Rest (binary)", fontsize=10,
                   color=C['text'])
    _centered_text(ax, s1_cx, s1_y + s1_h / 2 - 0.02,
                   "KW top-5 + Random Forest", fontsize=10,
                   color=C['text'])
    _centered_text(ax, s1_cx, s1_y + 0.035,
                   "500 trees, balanced weights", fontsize=9,
                   fontstyle='italic', color=C['text'])
    _centered_text(ax, s1_cx, s1_y + 0.012,
                   "FIXED across all pipelines", fontsize=9,
                   fontweight='bold', color=C['text'])

    # BA = 1.0 badge.
    badge_w = 0.06
    badge_h = 0.028
    badge_x = s1_x + s1_w - badge_w - 0.008
    badge_y = s1_y + s1_h - badge_h - 0.008
    _box(ax, badge_x, badge_y, badge_w, badge_h,
         fc=C['highlight'], ec='#C0A800', lw=1.0, pad=0.004)
    _centered_text(ax, badge_x + badge_w / 2, badge_y + badge_h / 2,
                   "BA=1.0", fontsize=8, fontweight='bold', color='#4A3F00')

    # Decision split - arrows from Stage 1 bottom centre.
    split_y = s1_y
    left_cx = col_l + bw * 0.18
    right_cx = col_cx

    # HER2+ output box (far left column area).
    her2_w = bw * 0.30
    her2_h = 0.04
    her2_y = 0.60
    her2_box_x = left_cx - her2_w / 2
    _box(ax, her2_box_x, her2_y, her2_w, her2_h,
         fc='#FADBD8', ec=C['HER2'], lw=1.0)
    _centered_text(ax, left_cx, her2_y + her2_h / 2,
                   "HER2+", fontsize=10, fontweight='bold', color=C['HER2'])

    # Left arrow: Stage 1 bottom -> HER2+ box.
    _arrow(ax, (col_cx - 0.035, split_y), (left_cx, her2_y + her2_h + 0.003),
           connectionstyle='arc3,rad=0.2', color=C['text'])

    # Right arrow: Stage 1 bottom -> Stage 2 top.
    s2_y = 0.37
    s2_h = 0.16
    _arrow(ax, (col_cx + 0.035, split_y), (right_cx, s2_y + s2_h + 0.003),
           connectionstyle='arc3,rad=-0.15', color=C['text'])

    # Horizontal labels beside the split arrows.
    _centered_text(ax, col_l + 0.035, 0.675,
                   "HER2+", fontsize=9, fontweight='bold',
                   color=C['HER2'],
                   bbox=dict(facecolor='white', edgecolor='none',
                             alpha=0.85, pad=1.5))
    _centered_text(ax, col_r - 0.03, 0.60,
                   "Rest", fontsize=10, fontweight='bold',
                   color=C['text'],
                   bbox=dict(facecolor='white', edgecolor='none',
                             alpha=0.85, pad=1.5))

    # Stage 2 box.
    s2_w = bw - 0.01
    s2_x = col_l + 0.005
    _box(ax, s2_x, s2_y, s2_w, s2_h, fc=C['stage2'], ec=C['stage2_dark'],
         lw=1.5)
    s2_cx = s2_x + s2_w / 2
    _centered_text(ax, s2_cx, s2_y + s2_h - 0.022,
                   "STAGE 2", fontsize=13, fontweight='bold',
                   color=C['text'])
    _centered_text(ax, s2_cx, s2_y + s2_h - 0.05,
                   "HR+ vs Triple Neg (binary)", fontsize=10,
                   color=C['text'])
    _centered_text(ax, s2_cx, s2_y + s2_h / 2 - 0.02,
                   "Feature selection + Classifier", fontsize=10,
                   color=C['text'])
    _centered_text(ax, s2_cx, s2_y + 0.035,
                   "TUNED via inner CV", fontsize=10,
                   fontweight='bold', color=C['text'])
    _centered_text(ax, s2_cx, s2_y + 0.012,
                   "4 pipeline variants compared", fontsize=10,
                   fontstyle='italic', color=C['text'])

    # Outputs from Stage 2.
    out_y = 0.25
    hr_x = col_l + bw * 0.3
    tn_x = col_l + bw * 0.7
    out_w = bw * 0.30
    out_h = 0.04

    # Arrows down from Stage 2 to HR+ and TN.
    _arrow(ax, (s2_cx - 0.035, s2_y), (hr_x, out_y + out_h + 0.003),
           connectionstyle='arc3,rad=0.1', color=C['text'])
    _arrow(ax, (s2_cx + 0.035, s2_y), (tn_x, out_y + out_h + 0.003),
           connectionstyle='arc3,rad=-0.1', color=C['text'])

    # HR+ box.
    _box(ax, hr_x - out_w / 2, out_y, out_w, out_h,
         fc='#D6E4F0', ec=C['HR'], lw=1.0)
    _centered_text(ax, hr_x, out_y + out_h / 2,
                   "HR+", fontsize=10, fontweight='bold', color=C['HR'])

    # TN box.
    _box(ax, tn_x - out_w / 2, out_y, out_w, out_h,
         fc='#D5F0EB', ec=C['TN'], lw=1.0)
    _centered_text(ax, tn_x, out_y + out_h / 2,
                   "Triple Neg", fontsize=10, fontweight='bold', color=C['TN'])

    # Final prediction box.
    final_y = 0.09
    final_w = bw * 0.75
    final_h = 0.055
    final_x = col_cx - final_w / 2
    _box(ax, final_x, final_y, final_w, final_h,
         fc='#F8F8F8', ec=C['border'], lw=1.5)
    _centered_text(ax, col_cx, final_y + final_h / 2,
                   "3-class prediction", fontsize=12, fontweight='bold')

    # Convergence arrows from all three output boxes to final box.
    # HER2+ arrow routes down through the gap between the gray CV box
    # (right edge x=0.63) and Stage 2 (left edge x=0.665), centred at ~0.648.
    _arrow(ax, (her2_box_x - 0.0097, her2_y + her2_h / 2),
           (final_x - 0.005, final_y + final_h / 2),
           connectionstyle='arc3,rad=0.13', lw=2, color=C['HER2'],
           mutation_scale=12, zorder=4)
    # HR+ straight down.
    _arrow(ax, (hr_x, out_y), (col_cx - 0.02, final_y + final_h + 0.003),
           connectionstyle='arc3,rad=0.05', lw=1.0, color=C['HR'],
           mutation_scale=12)
    # TN straight down.
    _arrow(ax, (tn_x, out_y), (col_cx + 0.02, final_y + final_h + 0.003),
           connectionstyle='arc3,rad=-0.05', lw=1.0, color=C['TN'],
           mutation_scale=12)


# ── Column dividers ─────────────────────────────────────────────────────────

def _draw_dividers(ax):
    """Draw subtle vertical divider lines between columns."""
    for x_div in [0.26, 0.645]:
        ax.plot([x_div, x_div], [0.02, 0.96],
                color='#CCCCCC', lw=0.8, ls=':', zorder=0)


# ── Panel labels ────────────────────────────────────────────────────────────

def _draw_panel_labels(ax):
    """Draw A, B, C panel labels in corners of each column."""
    labels = [
        (0.005, 1.016, "A"),
        (0.265, 1.016, "B"),
        (0.650, 1.016, "C"),
    ]
    for x, y, lbl in labels:
        ax.text(x, y, lbl, fontsize=16, fontweight='bold',
                color=C['text'], ha='left', va='top', zorder=10,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=1.0, pad=2.0))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Generate the slide 1 CV scheme diagram."""
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    _draw_dividers(ax)
    _draw_panel_labels(ax)
    _draw_column1(ax)
    _draw_column2(ax)
    _draw_column3(ax)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "slide_1_cv_scheme.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
