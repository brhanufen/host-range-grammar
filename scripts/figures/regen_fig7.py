#!/usr/bin/env python3
"""
Regenerate Figure 7 (mechanistic synthesis) with:
  - Colorblind-safe palette (replace red/green pairing)
  - Updated mechanism wording ("synergy" removed; "independent but functionally overlapping")
  - Single-panel layout (mechanism boxes only; former Panel B top-10 motif
    table removed as redundant with Table 1 and Figure 5B, co-author review)
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
    # Single-panel figure: the mechanistic schematic only. The former Panel B
    # (top-10 motif table) duplicated Table 1 (identical motifs/columns/values)
    # and its log2(E) column also appears in Figure 5B, so it was removed on
    # co-author review. Quantitative motif data lives in Table 1; the visual
    # enrichment ranking lives in Figure 5B.
    fig = plt.figure(figsize=(7.5, 2.9))
    axA = fig.add_subplot(1, 1, 1)
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

    # (No panel letter — this is now a single-panel figure.)

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
