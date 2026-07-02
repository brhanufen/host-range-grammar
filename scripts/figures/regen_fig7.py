#!/usr/bin/env python3
"""
Regenerate Figure 7 (mechanistic synthesis) with:
  - Colorblind-safe palette (replace red/green pairing)
  - Updated mechanism wording ("synergy" removed; "independent but functionally overlapping")
  - Same 2-panel layout (A: mechanism boxes; B: top-10 motif table)
"""

import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches

rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

import os
# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_fig7.py -> repo root is two directories up.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PKG  = os.path.dirname(_REPO)
OUT_MAIN = os.path.join(_REPO, 'figures')
OUT_REPO = os.path.join(_REPO, 'figures')

# Colorblind-safe palette: blue, purple, orange (no red/green confusion)
# Each box has a fill + edge colour at higher saturation
BOX_CPG   = {'fill': '#D9E7F5', 'edge': '#2171B5', 'title': '#1A4F7A'}  # blue
BOX_UPA   = {'fill': '#E8DDF0', 'edge': '#6A3D9A', 'title': '#3D1E5E'}  # purple
BOX_COMP  = {'fill': '#FCE5C9', 'edge': '#E6550D', 'title': '#9C3500'}  # orange

# Top 10 motifs (Group A reproducible — matches Table 1 in manuscript)
MOTIFS = [
    # (motif, log2E, cons_ARB, cons_ISFV, peak_pos, annotation)
    ('AUUAGG', '3.22', '0.60', '0.21', '0.78', 'UpA×1'),
    ('CCCUUU', '2.73', '0.90', '0.37', '0.38', '—'),
    ('AGAGAG', '2.59', '1.00', '0.94', '0.78', 'Purine-rich'),
    ('UACUCC', '2.57', '0.75', '0.33', '0.08', 'UpA×1'),
    ('UAGUAU', '2.51', '0.42', '0.20', '0.88', 'UpA×2; AU-rich'),
    ('AGAGGG', '2.43', '1.00', '0.81', '0.68', 'Purine-rich'),
    ('CUAAUA', '2.36', '0.51', '0.41', '0.93', 'UpA×2'),
    ('GGCCUC', '2.33', '0.91', '0.53', '0.33', 'GC-rich'),
    ('GAAUAG', '2.31', '0.83', '0.57', '0.18', 'UpA×1'),
    ('AUAGAA', '2.24', '0.86', '0.49', '0.73', 'UpA×1'),
]


def draw_box(ax, x, y, w, h, style, title, body):
    """Draw a rounded rectangle with title and body text."""
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle='round,pad=0.02,rounding_size=0.04',
                          linewidth=1.5, edgecolor=style['edge'],
                          facecolor=style['fill'])
    ax.add_patch(rect)
    # Title (centered, bold, larger)
    ax.text(x + w/2, y + h - 0.08, title, ha='center', va='top',
            fontsize=9, fontweight='bold', color=style['title'])
    # Body text
    ax.text(x + w/2, y + h - 0.22, body, ha='center', va='top',
            fontsize=7, color='#1a1a1a', wrap=True)


def make_figure7():
    fig = plt.figure(figsize=(7.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1.7], hspace=0.05)

    # ── Panel A: Mechanism boxes ──
    axA = fig.add_subplot(gs[0])
    axA.set_xlim(0, 1)
    axA.set_ylim(0, 1)
    axA.axis('off')

    # Section title
    axA.text(0.5, 0.97, 'Host Restriction Mechanisms Linked to Enriched Motifs',
             ha='center', va='top', fontsize=10, fontweight='bold')

    # Three boxes, side by side — box_y shifted down to give the section title clear space
    box_y = 0.05
    box_h = 0.75
    box_w = 0.28
    gap = 0.02

    # Box 1: CpG → ZAP / OAS3 (blue)
    cpg_body = ("ZAP binds CpG-enriched\n"
                "viral RNA; OAS3/RNaseL\n"
                "provides complementary\n"
                "restriction (independent\n"
                "but functionally\n"
                "overlapping pathways).\n\n"
                "No CpG-enriched motifs\n"
                "in top 10")
    draw_box(axA, 0.04, box_y, box_w, box_h, BOX_CPG,
             'CpG → ZAP / OAS3', cpg_body)

    # Box 2: UpA → ZAP / OAS3 (purple)
    upa_body = ("ZAP also binds UpA-enriched\n"
                "RNA; OAS3/RNaseL cleaves\n"
                "at UpA/UU sites. Mechanism\n"
                "of attenuation remains\n"
                "uncharacterized.\n\n"
                "6/10 motifs\n"
                "contain UpA")
    draw_box(axA, 0.04 + box_w + gap, box_y, box_w, box_h, BOX_UPA,
             'UpA → ZAP / OAS3', upa_body)

    # Box 3: Composition Bias (orange)
    comp_body = ("Purine-rich, GC-rich,\n"
                 "AU-rich patterns may\n"
                 "affect RNA structure\n"
                 "and stability.\n\n\n\n"
                 "3/10 motifs\n"
                 "purine/GC-rich")
    draw_box(axA, 0.04 + 2*(box_w + gap), box_y, box_w, box_h, BOX_COMP,
             'Composition Bias', comp_body)

    # Panel A label
    axA.text(-0.01, 1.02, 'A', transform=axA.transAxes,
             fontsize=14, fontweight='bold', va='top')

    # ── Panel B: Top 10 motif table ──
    axB = fig.add_subplot(gs[1])
    axB.set_xlim(0, 1)
    axB.set_ylim(0, 1)
    axB.axis('off')

    axB.text(0.5, 0.98, 'Top 10 Discriminative 6-mer Motifs',
             ha='center', va='top', fontsize=10, fontweight='bold')

    # Table data
    headers = ['Motif', r'log$_2$(E)', 'Cons.\nARB', 'Cons.\nISFV', 'Peak\nPos.', 'Biological Annotation']
    col_widths = [0.14, 0.10, 0.10, 0.10, 0.10, 0.30]
    col_starts = [0.08]
    for w in col_widths[:-1]:
        col_starts.append(col_starts[-1] + w)

    # Header row — taller than data rows to accommodate two-line headers
    # (Cons.\nARB, Cons.\nISFV, Peak\nPos.) without spilling over cell borders.
    header_y = 0.80
    header_h = 0.10   # taller for wrapped headers
    row_h = 0.07
    for i, (header, x_start, w) in enumerate(zip(headers, col_starts, col_widths)):
        # Header background (taller cell)
        axB.add_patch(patches.Rectangle((x_start, header_y - header_h), w, header_h,
                                         facecolor='#E5E5E5', edgecolor='#666666', linewidth=0.5))
        axB.text(x_start + w/2, header_y - header_h/2, header,
                 ha='center', va='center', fontsize=7, fontweight='bold',
                 linespacing=1.1)

    # Data rows — start immediately below the (taller) header row
    data_top = header_y - header_h
    for row_idx, row in enumerate(MOTIFS):
        y = data_top - row_h * (row_idx + 1)
        bg_color = '#FAFAFA' if row_idx % 2 == 0 else 'white'
        for col_idx, (cell, x_start, w) in enumerate(zip(row, col_starts, col_widths)):
            axB.add_patch(patches.Rectangle((x_start, y), w, row_h,
                                             facecolor=bg_color, edgecolor='#CCCCCC',
                                             linewidth=0.5))
            font_family = 'monospace' if col_idx == 0 else 'Arial'
            font_weight = 'bold' if col_idx == 0 else 'normal'
            axB.text(x_start + w/2, y + row_h/2, cell,
                     ha='center', va='center', fontsize=7,
                     fontfamily=font_family, fontweight=font_weight)

    axB.text(-0.01, 1.02, 'B', transform=axB.transAxes,
             fontsize=14, fontweight='bold', va='top')

    for fmt in ('png', 'pdf'):
        fig.savefig(f'{OUT_MAIN}/Figure7.{fmt}', dpi=600)
    plt.close()
    print('Figure 7 saved.')


if __name__ == '__main__':
    make_figure7()
    for fmt in ('png', 'pdf'):
        src = f'{OUT_MAIN}/Figure7.{fmt}'
        dst = f'{OUT_REPO}/Figure7.{fmt}'
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        print(f'  Copied Figure7.{fmt} -> reproducibility_repo/figures/')
    print('\nDone.')
