#!/usr/bin/env python3
"""Figure 4 — Genome-localized attribution and positional hotspots.

Fully reproducible and self-contained: all values are computed here from the
committed predictions file (no /tmp artifacts, no precomputed JSON required).

Source: data/windows_predictions_FIXED_SEED.csv
Method: class-stratified rank-based top-10% windows by P(own class) with
        deterministic tie-breaking (sort by ['p_arb','accession']), then
        head(int(N*0.10)); per-genome region check at boundary pos_rel=0.40.

Reproduces (verified):
  Panel B: ARB peak 1.8x at 0.38, ISFV peak 2.5x at 0.57
  Panel C: ARB Other 47% / NS3-NS5 65%, ISFV Other 35% / NS3-NS5 33%

Panel A: 6 P(ARB) tracks (3 ARB + 3 ISV)
Panel B: positional enrichment + data-aligned genome schematic underneath
Panel C: annotated genome schematic with percentages directly on regions
"""
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.transforms import blended_transform_factory
from scipy.ndimage import uniform_filter1d
import warnings; warnings.filterwarnings('ignore')

rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_fig4_repro.py -> repo root is two directories up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG  = os.path.dirname(REPO)
OUT_MAIN = os.path.join(REPO, 'figures')
OUT_REPO = os.path.join(REPO, 'figures')

# ── Load committed predictions ──────────────────────────────────────────
df = pd.read_csv(os.path.join(REPO, 'data', 'windows_predictions_FIXED_SEED.csv'))
BOUNDARY = 0.40

# ── Class-stratified rank-based top-10% (deterministic tie-break) ────────
n_bins = 20
arb_w = df[df.label == 'ARB']
isv_w = df[df.label == 'ISV']
arb_sorted = arb_w.sort_values(['p_arb', 'accession'], ascending=[False, True])
isv_sorted = isv_w.sort_values(['p_arb', 'accession'], ascending=[True, True])
arb_top = arb_sorted.head(int(len(arb_w) * 0.10))
isv_top = isv_sorted.head(int(len(isv_w) * 0.10))

# Panel B: positional enrichment (observed / expected) per bin
arb_enrichment, isv_enrichment = [], []
for b in range(n_bins):
    bs, be = b / n_bins, (b + 1) / n_bins
    all_arb = ((arb_w.pos_rel >= bs) & (arb_w.pos_rel < be)).sum()
    top_arb = ((arb_top.pos_rel >= bs) & (arb_top.pos_rel < be)).sum()
    arb_enrichment.append((top_arb / len(arb_top)) / (all_arb / len(arb_w)) if all_arb > 0 else 0)
    all_isv = ((isv_w.pos_rel >= bs) & (isv_w.pos_rel < be)).sum()
    top_isv = ((isv_top.pos_rel >= bs) & (isv_top.pos_rel < be)).sum()
    isv_enrichment.append((top_isv / len(isv_top)) / (all_isv / len(isv_w)) if all_isv > 0 else 0)
arb_enrichment = np.array(arb_enrichment)
isv_enrichment = np.array(isv_enrichment)
bin_centers = np.array([(b + 0.5) / n_bins for b in range(n_bins)])

# Panel C: per-genome regional enrichment (% with >=1 top window in region)
def region_pcts(top, w):
    ntot = w['accession'].nunique()
    other = top[top.pos_rel < BOUNDARY]['accession'].nunique()
    ns35 = top[top.pos_rel >= BOUNDARY]['accession'].nunique()
    return other / ntot * 100, ns35 / ntot * 100, ntot
ARB_OTHER, ARB_NS35, ARB_N = region_pcts(arb_top, arb_w)
ISV_OTHER, ISV_NS35, ISV_N = region_pcts(isv_top, isv_w)

C_ARB = '#2171B5'
C_ISV = '#E6550D'

# Polyprotein layout (DENV-2 NC_001474 proportions)
poly_proteins = [
    ('5′\nUTR', 0.000, 0.009, '#FFFFFF'),
    ('C',    0.009, 0.041, '#FCE5C9'),
    ('prM',  0.041, 0.087, '#FCE5C9'),
    ('E',    0.087, 0.226, '#FCE5C9'),
    ('NS1',  0.226, 0.324, '#D9E7F5'),
    ('NS2A', 0.324, 0.385, '#C6DBEF'),
    ('NS2B', 0.385, 0.421, '#9ECAE1'),
    ('NS3',  0.421, 0.594, '#FFB347'),
    ('NS4A', 0.594, 0.630, '#FFAA33'),
    ('NS4B', 0.630, 0.700, '#FF9933'),
    ('NS5',  0.700, 0.958, '#FF8800'),
    ('3′\nUTR', 0.958, 1.000, '#FFFFFF'),
]

def draw_schematic_in_axes(ax, y0, h, fontsize=5.5):
    """Draw flavivirus genome schematic spanning x=[0,1] inside given axes."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for name, x0, x1, color in poly_proteins:
        rect = mpatches.Rectangle((x0, y0), x1 - x0, h, transform=trans,
                                  facecolor=color, edgecolor='#444444',
                                  linewidth=0.6, clip_on=False)
        ax.add_patch(rect)
        mid = (x0 + x1) / 2
        if 'UTR' in name:
            label = name.replace('\n', ' ')
            if x0 < 0.5:
                ax.text(0.0, y0 - 0.05, label, transform=trans,
                        ha='left', va='top', fontsize=fontsize - 1.5,
                        color='#333333', clip_on=False)
            else:
                ax.text(1.0, y0 - 0.05, label, transform=trans,
                        ha='right', va='top', fontsize=fontsize - 1.5,
                        color='#333333', clip_on=False)
            ax.plot([mid, mid], [y0, y0 - 0.035], transform=trans,
                    color='#666666', lw=0.6, clip_on=False)
        else:
            font = fontsize if (x1 - x0) > 0.04 else fontsize - 1
            ax.text(mid, y0 + h/2, name, transform=trans,
                    ha='center', va='center', fontsize=font, color='black')

# === Build figure ===
fig = plt.figure(figsize=(7.5, 9.5))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1.6, 1.8],
                     hspace=0.70, wspace=0.32)

# ───── Panel A: 6 P(ARB) tracks ─────
target_genomes = [
    ('AY898809', 'Alfuy virus', 'ARB'),
    ('MG182017', 'Spondweni virus', 'ARB'),
    ('MK007532', 'Louping ill virus', 'ARB'),
    ('NC_040610', 'Nanay virus', 'ISFV'),
    ('JQ308185', 'Chaoyang virus (HLD115)', 'ISFV'),
    ('MW246770', 'Chaoyang virus (NM)', 'ISFV'),
]
axA0 = None
for i, (acc, name, label) in enumerate(target_genomes):
    row, col = i // 3, i % 3
    ax = fig.add_subplot(gs[row, col])
    if i == 0:
        axA0 = ax
    genome = df[df.accession == acc].sort_values('pos_rel')
    if len(genome) == 0:
        continue
    color = C_ARB if label == 'ARB' else C_ISV
    ax.plot(genome.pos_rel, genome.p_arb, '-', color=color, alpha=0.3, lw=0.6)
    smoothed = uniform_filter1d(genome.p_arb.values, size=3)
    ax.plot(genome.pos_rel, smoothed, '-', color=color, lw=1.6)
    ax.set_title(f'{name}\n({acc}, {label})', fontsize=6.5, pad=2)
    ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    if col == 0: ax.set_ylabel('P(ARB)', fontsize=7)
    if row == 1: ax.set_xlabel('Relative position', fontsize=7)

# Line-type legend on the first Panel A subplot: light = raw per-window P(ARB),
# bold = 3-window smoothed (reviewer comment: needs a legend for light vs solid lines).
if axA0 is not None:
    from matplotlib.lines import Line2D
    _linelegend = [
        Line2D([0], [0], color='#555555', lw=0.6, alpha=0.4, label='Per-window P(ARB)'),
        Line2D([0], [0], color='#555555', lw=1.6, label='Smoothed (3-window mean)'),
    ]
    axA0.legend(handles=_linelegend, loc='upper left', fontsize=5.0,
                frameon=True, framealpha=0.85, handlelength=1.6,
                handletextpad=0.4, borderpad=0.3, labelspacing=0.25)

# ───── Panel B: positional enrichment + data-aligned schematic ─────
ax_b = fig.add_subplot(gs[2, :])
ax_b.axvspan(BOUNDARY, 1.0, alpha=0.10, color='#FFB347', zorder=0)
ax_b.axvline(BOUNDARY, color='#888888', ls='--', lw=0.7, zorder=1)

arb_peak_i = int(np.argmax(arb_enrichment))
isv_peak_i = int(np.argmax(isv_enrichment))
ax_b.plot(bin_centers, arb_enrichment, color=C_ARB, lw=1.8, marker='o', markersize=4,
          label=f'ARB (peak {arb_enrichment[arb_peak_i]:.1f}× at {bin_centers[arb_peak_i]:.2f})', zorder=3)
ax_b.plot(bin_centers, isv_enrichment, color=C_ISV, lw=1.8, marker='s', markersize=4,
          label=f'ISFV (peak {isv_enrichment[isv_peak_i]:.1f}× at {bin_centers[isv_peak_i]:.2f})', zorder=3)
ax_b.axhline(1.0, color='gray', ls='--', lw=0.6, zorder=1)
ax_b.set_xlim(0, 1)
ax_b.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax_b.set_ylabel('Enrichment\n(observed / expected)', fontsize=8)
ax_b.set_xlabel('Relative genome position', fontsize=8, labelpad=2)
ax_b.legend(loc='upper right', fontsize=6.5)
ax_b.set_title('Positional enrichment of top-10% windows', fontsize=9)
draw_schematic_in_axes(ax_b, y0=-0.38, h=0.10, fontsize=5.5)

# ───── Panel C: annotated genome schematic (percentages on schematic) ─────
ax_c = fig.add_subplot(gs[3, :])
ax_c.set_xlim(0, 1)
ax_c.set_ylim(0, 1)
ax_c.axis('off')

sch_y = 0.42
sch_h = 0.18

ax_c.add_patch(mpatches.Rectangle((0.0, sch_y - 0.05), BOUNDARY, sch_h + 0.10,
                                   facecolor='#E0E0E0', alpha=0.40, zorder=0))
ax_c.add_patch(mpatches.Rectangle((BOUNDARY, sch_y - 0.05), 1 - BOUNDARY, sch_h + 0.10,
                                   facecolor='#FFB347', alpha=0.20, zorder=0))

for name, x0, x1, color in poly_proteins:
    rect = mpatches.Rectangle((x0, sch_y), x1 - x0, sch_h,
                              facecolor=color, edgecolor='#333333', linewidth=0.5, zorder=2)
    ax_c.add_patch(rect)
    mid = (x0 + x1) / 2
    width = x1 - x0
    if 'UTR' in name:
        label = name.replace('\n', ' ')
        if x0 < 0.5:
            ax_c.text(0.0, sch_y - 0.04, label, ha='left', va='top',
                      fontsize=5.5, color='#333333', zorder=3)
        else:
            ax_c.text(1.0, sch_y - 0.04, label, ha='right', va='top',
                      fontsize=5.5, color='#333333', zorder=3)
        ax_c.plot([mid, mid], [sch_y, sch_y - 0.03],
                  color='#666666', lw=0.5, zorder=2)
    elif width > 0.045:
        ax_c.text(mid, sch_y + sch_h/2, name, ha='center', va='center',
                  fontsize=7, color='black', zorder=3)
    else:
        ax_c.text(mid, sch_y + sch_h/2, name, ha='center', va='center',
                  fontsize=5, color='black', zorder=3)

ax_c.axvline(BOUNDARY, ymin=0.25, ymax=0.82, color='#444444', ls='--', lw=1.0, zorder=1)

ax_c.text(BOUNDARY/2, sch_y + sch_h + 0.06, 'Other (5′UTR–NS2B)',
          ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#444444')
ax_c.text((BOUNDARY+1)/2, sch_y + sch_h + 0.06, 'NS3–NS5 region',
          ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#A85800')

ax_c.text(BOUNDARY/2, sch_y + sch_h + 0.18,
          f'ARB: {ARB_OTHER:.0f}%', ha='center', va='bottom',
          fontsize=11, fontweight='bold', color=C_ARB)
ax_c.text((BOUNDARY+1)/2, sch_y + sch_h + 0.18,
          f'ARB: {ARB_NS35:.0f}%', ha='center', va='bottom',
          fontsize=11, fontweight='bold', color=C_ARB)

ax_c.text(BOUNDARY/2, sch_y - 0.12,
          f'ISFV: {ISV_OTHER:.0f}%', ha='center', va='top',
          fontsize=11, fontweight='bold', color=C_ISV)
ax_c.text((BOUNDARY+1)/2, sch_y - 0.12,
          f'ISFV: {ISV_NS35:.0f}%', ha='center', va='top',
          fontsize=11, fontweight='bold', color=C_ISV)

ax_c.text(0.5, 0.99,
          'Per-genome regional enrichment: % of genomes with ≥1 top-10% window in each region',
          ha='center', va='top', fontsize=9, fontweight='bold')
ax_c.text(0.5, 0.06,
          f"Class-stratified rank-based top-10% windows by P(own class); "
          f"ARB n={ARB_N}, ISFV n={ISV_N}; SEED=42 reproducible.",
          ha='center', va='top', fontsize=6.5, color='#666666', style='italic')

# ── Panel letters ──
for _axp, _ltr in [(axA0, 'A'), (ax_b, 'B')]:
    _p = _axp.get_position()
    fig.text(_p.x0 - 0.045, _p.y1 + 0.012, _ltr,
             fontsize=14, fontweight='bold', va='bottom', ha='left')
ax_c.text(-0.05, sch_y + sch_h / 2, 'C', transform=ax_c.transAxes,
          fontsize=14, fontweight='bold', va='center', ha='left', clip_on=False)

# Save
os.makedirs(OUT_MAIN, exist_ok=True)
os.makedirs(OUT_REPO, exist_ok=True)
for fmt in ('png', 'pdf'):
    out = os.path.join(OUT_MAIN, f'Figure4.{fmt}')
    fig.savefig(out, dpi=600)
    print(f'  Saved: {out}')
    _dst = os.path.join(OUT_REPO, f'Figure4.{fmt}')
    if os.path.abspath(out) != os.path.abspath(_dst):
        shutil.copy(out, _dst)
plt.close()

print('\nValues used (reproducible, computed inline from windows_predictions_FIXED_SEED.csv):')
print(f'  Panel B: ARB {arb_enrichment[arb_peak_i]:.2f}× at {bin_centers[arb_peak_i]:.2f}    '
      f'ISFV {isv_enrichment[isv_peak_i]:.2f}× at {bin_centers[isv_peak_i]:.2f}')
print(f'  Panel C: ARB {ARB_OTHER:.1f}%/{ARB_NS35:.1f}%   ISFV {ISV_OTHER:.1f}%/{ISV_NS35:.1f}%')
