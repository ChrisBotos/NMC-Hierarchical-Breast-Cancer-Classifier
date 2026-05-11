"""
Group 9.
Authors:
    Alexandros Michailidis (2903034).
    Antonie Wagner (2903383).
    Christos Botos (2878553).
    Yan Qiao (2874296).
Affiliation: Computer Science and Bioinformatics Master's Programmes.

Script Name: make_slide3_methods_flow.py.
Description:
    Generates a horizontal flowchart figure showing the full submitted method
    (EN+NMC plateau hierarchical classifier) end-to-end for presentation
    slide 3. The flowchart covers: raw data, region merging, Stage 1
    (HER2+ vs Rest), routing split, Stage 2 (HR+ vs TN), and final
    3-class output.

Usage:
    python3 presentation/make_slide3_methods_flow.py

Dependencies:
    Python >= 3.10.
    matplotlib.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

# --- Colour palette. ---
C = {
    'stage1': '#4DBBD5',
    'stage1_dark': '#2D7A8E',
    'stage2': '#F39B7F',
    'stage2_dark': '#C06040',
    'preproc': '#91D1C2',
    'preproc_dark': '#5DA08E',
    'data': '#E8E8E8',
    'highlight': '#FFD700',
    'border': '#444444',
    'text': '#1A1A1A',
    'arrow': '#1A1A1A',
    'HER2': '#E64B35',
    'HR': '#3C5488',
    'TN': '#00A087',
    'output': '#B09C85',
}


def box(ax, x, y, w, h, fc, ec, lw=1.5, zorder=2):
    """Draw a rounded rectangle and return the patch.

    Args:
        ax (Axes): Matplotlib axes.
        x (float): Left edge in axes coordinates.
        y (float): Bottom edge in axes coordinates.
        w (float): Width.
        h (float): Height.
        fc (str): Fill colour.
        ec (str): Border colour.
        lw (float): Border width.
        zorder (int): Drawing order.

    Returns:
        FancyBboxPatch: The created patch.
    """
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008",
        facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(p)
    return p


def arr(ax, x0, y0, x1, y1, color=None, lw=2.0, rad=0.0, zorder=3):
    """Draw a straight or curved arrow between two points.

    Args:
        ax (Axes): Matplotlib axes.
        x0 (float): Start x.
        y0 (float): Start y.
        x1 (float): End x.
        y1 (float): End y.
        color (str): Arrow colour.
        lw (float): Line width.
        rad (float): Arc curvature radius.
        zorder (int): Drawing order.

    Returns:
        FancyArrowPatch: The created arrow.
    """
    if color is None:
        color = C['arrow']
    a = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='->,head_length=6,head_width=4',
        mutation_scale=1, linewidth=lw, color=color,
        zorder=zorder, transform=ax.transAxes,
        clip_on=False,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(a)
    return a


def label_box(ax, x, y, w, h, text, color, fs=11):
    """Draw a small coloured label box with centred text.

    Args:
        ax (Axes): Matplotlib axes.
        x (float): Left edge.
        y (float): Bottom edge.
        w (float): Width.
        h (float): Height.
        text (str): Label text.
        color (str): Background colour.
        fs (int): Font size.
    """
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005",
        facecolor=color, edgecolor=C['border'],
        linewidth=1.2, zorder=4,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text,
            ha='center', va='center',
            fontsize=fs, fontweight='bold', color='white',
            transform=ax.transAxes, zorder=5)


def txt(ax, x, y, s, **kw):
    """Shorthand for axes-coordinate text with consistent defaults.

    Args:
        ax (Axes): Matplotlib axes.
        x (float): X position.
        y (float): Y position.
        s (str): Text string.
        **kw: Additional kwargs passed to ax.text.
    """
    defaults = dict(
        ha='center', va='center', color=C['text'],
        transform=ax.transAxes, zorder=5,
    )
    defaults.update(kw)
    ax.text(x, y, s, **defaults)


def main():
    """Generate the slide 3 methods flowchart figure."""
    fig, ax = plt.subplots(figsize=(13.33, 7.5))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # ------------------------------------------------------------------ #
    # Layout grid.                                                        #
    # The main flow line sits at cy=0.42 so HER2+ branch has room above. #
    # ------------------------------------------------------------------ #
    cy = 0.42
    bh = 0.32
    bt = cy + bh / 2
    bb = cy - bh / 2

    # Column positions: (left_edge, width).
    x1, w1 = 0.01, 0.085           # Raw Data.
    x2, w2 = 0.125, 0.095          # Region Merging.
    x3, w3 = 0.255, 0.185          # Stage 1 (wider for title text).
    dx     = 0.475                  # Diamond x-centre.
    x4, w4 = 0.53, 0.175           # Stage 2.
    x5, w5 = 0.83, 0.16            # Output.

    g = 0.012                       # Arrow gap from box edge.

    # ================================================================== #
    # BOX 1 - Raw aCGH Data.                                             #
    # ================================================================== #
    box(ax, x1, bb, w1, bh, C['data'], C['border'])
    txt(ax, x1 + w1 / 2, bt - 0.03, "Raw aCGH\nData",
        fontsize=13, fontweight='bold', va='top')
    for i, s in enumerate(["100 samples", "2,834 regions", "CN: {-1,0,1,2}"]):
        txt(ax, x1 + w1 / 2, cy - 0.02 - i * 0.045, s, fontsize=10)

    # ================================================================== #
    # ARROW 1 -> 2.                                                      #
    # ================================================================== #
    arr(ax, x1 + w1 + g, cy, x2 - g, cy)

    # ================================================================== #
    # BOX 2 - Region Merging.                                            #
    # ================================================================== #
    box(ax, x2, bb, w2, bh, C['preproc'], C['preproc_dark'])
    txt(ax, x2 + w2 / 2, bt - 0.03, "Region\nMerging",
        fontsize=13, fontweight='bold', va='top')
    for i, s in enumerate(["Label-free", "Pearson r > 0.8", "Adjacent collapse"]):
        txt(ax, x2 + w2 / 2, cy - 0.02 - i * 0.045, s, fontsize=10)
    txt(ax, x2 + w2 / 2, bb - 0.025, "2,834 --> 273",
        fontsize=11, fontweight='bold', color=C['preproc_dark'], va='top')

    # ================================================================== #
    # ARROW 2 -> 3.                                                      #
    # ================================================================== #
    arr(ax, x2 + w2 + g, cy, x3 - g, cy)

    # ================================================================== #
    # BOX 3 - Stage 1.                                                   #
    # ================================================================== #
    box(ax, x3, bb, w3, bh, C['stage1'], C['stage1_dark'], lw=2)
    txt(ax, x3 + w3 / 2, bt - 0.02, "Stage 1:\nHER2+ vs Rest",
        fontsize=12, fontweight='bold', va='top')
    for i, s in enumerate([
        "KW top-5 + RF (500 trees)",
        "Balanced weights",
        "FIXED (no tuning)",
    ]):
        txt(ax, x3 + w3 / 2, cy + 0.005 - i * 0.045, s, fontsize=10)

    # Gold BA=1.0 badge.
    bx3c = x3 + w3 / 2
    bby = cy - 0.13
    bbw, bbh = 0.058, 0.035
    box(ax, bx3c - bbw / 2, bby - bbh / 2, bbw, bbh,
        C['highlight'], C['border'], lw=1.2, zorder=4)
    txt(ax, bx3c, bby, "BA = 1.0", fontsize=9.5, fontweight='bold')

    # Below-box annotation.
    txt(ax, x3 + w3 / 2, bb - 0.02, "5 chr17 features (ERBB2)",
        fontsize=11, fontweight='bold', color=C['stage1_dark'], va='top')

    # ================================================================== #
    # ARROW Stage 1 -> diamond.                                          #
    # ================================================================== #
    arr(ax, x3 + w3 + g, cy, dx - 0.015, cy)

    # ================================================================== #
    # DECISION DIAMOND.                                                  #
    # ================================================================== #
    ds = 0.022
    diamond = Polygon(
        [(dx, cy + ds), (dx + ds * 0.7, cy),
         (dx, cy - ds), (dx - ds * 0.7, cy), (dx, cy + ds)],
        closed=True, facecolor='white', edgecolor=C['border'],
        linewidth=1.5, zorder=6,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(diamond)

    # ================================================================== #
    # UPPER BRANCH - HER2+.                                              #
    # ================================================================== #
    her2_w, her2_h = 0.06, 0.048
    her2_x = dx - her2_w / 2
    her2_y = 0.83

    arr(ax, dx, cy + ds + 0.002, dx, her2_y - 0.002,
        color=C['HER2'], lw=1.8)
    label_box(ax, her2_x, her2_y, her2_w, her2_h, "HER2+", C['HER2'], fs=12)
    txt(ax, dx - 0.04, 0.72, "Predicted\nHER2+",
        fontsize=10, color=C['HER2'], ha='right')

    # ================================================================== #
    # LOWER BRANCH - Rest -> Stage 2.                                    #
    # ================================================================== #
    arr(ax, dx + ds * 0.7 + 0.002, cy, x4 - g, cy, lw=2.0)
    txt(ax, dx + 0.025, cy + 0.04, "Rest",
        fontsize=11, fontweight='bold', color=C['text'], va='bottom')

    # ================================================================== #
    # BOX 4 - Stage 2.                                                   #
    # ================================================================== #
    box(ax, x4, bb, w4, bh, C['stage2'], C['stage2_dark'], lw=2)
    txt(ax, x4 + w4 / 2, bt - 0.02, "Stage 2:\nHR+ vs TN",
        fontsize=12, fontweight='bold', va='top')
    for i, s in enumerate([
        "Elastic Net selector",
        "C=0.046, l1=0.1, k=50",
        "NMC (no shrinkage)",
        "Plateau ensemble:",
        "top 15 configs pooled",
    ]):
        txt(ax, x4 + w4 / 2, cy + 0.02 - i * 0.04, s, fontsize=10)

    # Below-box annotations.
    txt(ax, x4 + w4 / 2, bb - 0.02, "~82 features/fold",
        fontsize=11, fontweight='bold', color=C['stage2_dark'], va='top')
    txt(ax, x4 + w4 / 2, bb - 0.05,
        "chr6p, chr12q, chr5q, chr16, chr22",
        fontsize=10, fontweight='bold', fontstyle='italic',
        color=C['stage2_dark'], va='top')

    # ================================================================== #
    # CLASS LABEL BOXES from Stage 2.                                    #
    # ================================================================== #
    lw2, lh2 = 0.042, 0.042

    hr_x = x4 + w4 + 0.04
    hr_y = cy + 0.06
    arr(ax, x4 + w4 + g, cy + 0.03, hr_x, hr_y + lh2 / 2,
        color=C['HR'], lw=1.5)
    label_box(ax, hr_x, hr_y, lw2, lh2, "HR+", C['HR'], fs=11)

    tn_x = x4 + w4 + 0.04
    tn_y = cy - 0.10
    arr(ax, x4 + w4 + g, cy - 0.03, tn_x, tn_y + lh2 / 2,
        color=C['TN'], lw=1.5)
    label_box(ax, tn_x, tn_y, lw2, lh2, "TN", C['TN'], fs=11)

    # ================================================================== #
    # BOX 5 - Output.                                                    #
    # ================================================================== #
    oh = 0.34
    ob = cy - oh / 2
    box(ax, x5, ob, w5, oh, C['output'], C['border'], lw=1.5)
    txt(ax, x5 + w5 / 2, cy + oh / 2 - 0.025, "3-Class Prediction",
        fontsize=13, fontweight='bold', va='top', color='white')

    # Three small coloured dots/boxes inside output to show classes.
    cdot_x = x5 + 0.015
    cdot_w = 0.012
    for i, (col, lab) in enumerate([
        (C['HER2'], "HER2+"),
        (C['HR'], "HR+"),
        (C['TN'], "Triple Neg"),
    ]):
        cdot_y = cy + 0.04 - i * 0.045
        box(ax, cdot_x, cdot_y - 0.01, cdot_w, 0.02,
            col, C['border'], lw=0.6, zorder=4)
        txt(ax, cdot_x + cdot_w + 0.01, cdot_y, lab,
            fontsize=10, color='white', ha='left', fontweight='bold')

    # Estimate text at bottom of output box.
    txt(ax, x5 + w5 / 2, cy - 0.075, "Estimate:",
        fontsize=10.5, fontweight='bold', color='white')
    txt(ax, x5 + w5 / 2, cy - 0.11, "48/57 correct (~84.2%)",
        fontsize=10, color='white')

    # ================================================================== #
    # CONVERGING ARROWS to output box.                                   #
    # ================================================================== #
    # From HER2+ box right edge, sweeping right to output top.
    arr(ax, her2_x + her2_w, her2_y + her2_h / 2,
        x5 + w5 / 2, cy + oh / 2 + 0.005,
        color=C['HER2'], lw=1.5, rad=-0.25)
    # From HR+ to output left edge.
    arr(ax, hr_x + lw2, hr_y + lh2 / 2,
        x5, cy + 0.04,
        color=C['HR'], lw=1.5)
    # From TN to output left edge.
    arr(ax, tn_x + lw2, tn_y + lh2 / 2,
        x5, cy - 0.04,
        color=C['TN'], lw=1.5)

    # ================================================================== #
    # DIMENSION ANNOTATIONS along the top.                               #
    # ================================================================== #
    ty = bt + 0.04
    for xx, s in [
        (x1 + w1 / 2, "n=100, p=2,834"),
        (x2 + w2 / 2, "n=100, p=273"),
        (x3 + w3 / 2, "k=5 features"),
        (x4 + w4 / 2, "~82 features"),
    ]:
        txt(ax, xx, ty, s, fontsize=11, fontweight='bold', color='#555555',
            fontstyle='italic', va='bottom')

    # ================================================================== #
    # LEGEND - bottom left.                                              #
    # ================================================================== #
    for i, (col, lab, tc) in enumerate([
        (C['stage2'], "Stage 2 (HR+ vs TN)", C['stage2_dark']),
        (C['stage1'], "Stage 1 (HER2+ vs Rest)", C['stage1_dark']),
        (C['preproc'], "Preprocessing", C['preproc_dark']),
    ]):
        ly = 0.035 + i * 0.05
        box(ax, 0.01, ly, 0.018, 0.025, col, C['border'], lw=0.8, zorder=4)
        txt(ax, 0.038, ly + 0.012, lab,
            fontsize=11, fontweight='bold', ha='left', color=tc)

    # ================================================================== #
    # Save.                                                              #
    # ================================================================== #
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "slide_3_methods_flow.png",
    )
    fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
