#!/usr/bin/env python3
"""
Regenerate Figure 2 (classification performance) for the flavivirus manuscript.

Three panels:
  A: PR curves for DNABERT-2 under Random / Accession-grouped / Species-grouped CV
  B: PR curves for all methods under Species-grouped CV
  C: Bar chart of AP for all methods × Accession/Species + Position-only chance

Cosmetic fix vs prior version: Panel C value labels for species-grouped bars are
placed ABOVE the error bar whiskers (not inside the bars), matching Fig 6B style.
"""

import os
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import precision_recall_curve, average_precision_score

# Publication style — matches regen_fig5_fig6.py
rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 6.5,
    'legend.frameon': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

C_RAND   = '#969696'
C_ACC    = '#2171B5'
C_SP     = '#E6550D'
C_CHANCE = '#BDBDBD'
# Panel B (methods) palette — deliberately distinct from the CV-scheme
# blue/orange used in Panels A and C, so a colour never means two things
# across the figure (per co-author review). Methods form their own family:
# purple / teal / green.
C_DNABERT = '#6A51A3'  # purple (was #2171B5 — collided with Accession-grouped)
C_TFIDF   = '#1B9E9E'  # teal   (was #E6550D — collided with Species-grouped)
C_COMP    = '#31A354'  # green  (unchanged — already distinct)
C_MOTIF   = '#756BB1'

# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_fig2.py -> repo root is two directories up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG  = os.path.dirname(REPO)
OUT_MAIN = os.path.join(REPO, 'figures')
OUT_REPO = os.path.join(REPO, 'figures')

# Load data
data = np.load(os.path.join(REPO, 'results', 'oof_predictions.npz'))
y = data['y_isv']  # 1 = ISFV (positive class), 0 = ARB

with open(os.path.join(REPO, 'results', 'all_baseline_results.json')) as f:
    res = json.load(f)

# Chance level = positive class fraction
chance = y.mean()


def pr_curve(scores):
    """Returns (precision, recall, ap) using ISFV as positive class.
    The stored OOF scores are already P(ISFV) (positive-class probabilities)
    — verified by mean(scores | ISFV)≈0.94, mean(scores | ARB)≈0.05.
    Note: the manuscript Methods refers to predictions as P(ARB) for the
    attribution panels — for the PR analysis here, we use ISFV-as-positive."""
    precision, recall, _ = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    return precision, recall, ap


def make_figure2():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.3),
                              gridspec_kw={'height_ratios': [1, 1],
                                           'wspace': 0.32, 'hspace': 0.45})
    # Layout: A and B on top row, C spans bottom row
    axA = axes[0, 0]
    axB = axes[0, 1]
    # Remove bottom row axes — we'll add a single wider axes for C
    axes[1, 0].remove()
    axes[1, 1].remove()
    axC = fig.add_subplot(2, 1, 2)

    # ═══════ Panel A: DNABERT-2 PR curves under 3 CV schemes ═══════
    # PR curves drawn from pooled OOF predictions across folds (visual aid).
    # AP values in labels are the per-fold MEAN AP reported in Methods/Results
    # (from JSON), matching the Panel C bar chart and the manuscript text.
    for scheme, key, color, label, ap in [
        ('random',     'dnabert2_rand', C_RAND, 'Random CV',         res['DNABERT-2']['random']['ap_mean']),
        ('accession',  'dnabert2_acc',  C_ACC,  'Accession-grouped', res['DNABERT-2']['accession']['ap_mean']),
        ('species',    'dnabert2_sp',   C_SP,   'Species-grouped',   res['DNABERT-2']['species']['ap_mean']),
    ]:
        p, r, _ = pr_curve(data[key])
        axA.plot(r, p, color=color, lw=1.2,
                 label=f'{label} (AP = {ap:.3f})')
    axA.axhline(chance, color=C_CHANCE, lw=0.7, ls=':', zorder=0)
    axA.text(0.97, chance + 0.02, f'Chance ({chance:.3f})',
             fontsize=5.5, color='#666666', ha='right')
    axA.set_xlabel('Recall')
    axA.set_ylabel('Precision')
    axA.set_xlim([0, 1])
    axA.set_ylim([0, 1.05])
    axA.legend(loc='lower left', fontsize=5.5, handletextpad=0.5, labelspacing=0.6)
    axA.set_title('DNABERT-2:\nEffect of Evaluation Scheme',
                  fontweight='bold', fontsize=8, pad=4)

    # ═══════ Panel B: All methods PR under Species-grouped CV ═══════
    # Same convention as Panel A — pooled visual, per-fold mean AP in labels.
    for key, color, label, ap in [
        ('dnabert2_sp', C_DNABERT, 'DNABERT-2',    res['DNABERT-2']['species']['ap_mean']),
        ('tfidf_sp',    C_TFIDF,   'k-mer TF-IDF', res['k-mer TF-IDF']['species']['ap_mean']),
        ('comp_sp',     C_COMP,    'Composition',  res['Composition']['species']['ap_mean']),
    ]:
        p, r, _ = pr_curve(data[key])
        axB.plot(r, p, color=color, lw=1.2,
                 label=f'{label} (AP = {ap:.3f})')
    # The position-only baseline performs at exactly chance (AUC = 0.5,
    # AP = 0.071 = ISFV base rate), so it coincides with this line rather
    # than forming an informative curve — labelled explicitly per co-author
    # review so the negative control is represented in this panel too.
    axB.axhline(chance, color=C_CHANCE, lw=0.7, ls=':', zorder=0)
    axB.text(0.97, chance + 0.02, f'Position-only \u2248 Chance ({chance:.3f})',
             fontsize=5.5, color='#666666', ha='right')
    axB.set_xlabel('Recall')
    axB.set_ylabel('Precision')
    axB.set_xlim([0, 1])
    axB.set_ylim([0, 1.05])
    axB.legend(loc='lower left', fontsize=5.5, handletextpad=0.5, labelspacing=0.6)
    axB.set_title('All Methods Under\nSpecies-Grouped CV',
                  fontweight='bold', fontsize=8, pad=4)

    # ═══════ Panel C: AP bar chart (all methods x Accession/Species + Position-only) ═══════
    methods = ['DNABERT-2', 'k-mer\nTF-IDF', 'Compo-\nsition', 'Position-\nonly']
    method_keys = ['DNABERT-2', 'k-mer TF-IDF', 'Composition', 'Position-only']

    ap_acc_vals = [res[k]['accession']['ap_mean'] for k in method_keys]
    ap_sp_vals  = [res[k]['species']['ap_mean']   for k in method_keys]
    sd_acc_vals = [res[k]['accession']['ap_std']   for k in method_keys]
    sd_sp_vals  = [res[k]['species']['ap_std']     for k in method_keys]

    x = np.arange(len(methods))
    w = 0.32
    bars_a = axC.bar(x - w/2, ap_acc_vals, w, yerr=sd_acc_vals, color=C_ACC,
                      edgecolor='white', linewidth=0.5,
                      error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8},
                      label='Accession-grouped', zorder=3)
    bars_s = axC.bar(x + w/2, ap_sp_vals, w, yerr=sd_sp_vals, color=C_SP,
                      edgecolor='white', linewidth=0.5,
                      error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8},
                      label='Species-grouped', zorder=3)

    # Value labels — ALWAYS above the error bar whisker, matching Fig 6B style.
    for i, (bar, val) in enumerate(zip(bars_a, ap_acc_vals)):
        y_top = val + sd_acc_vals[i] + 0.02
        axC.text(bar.get_x() + bar.get_width()/2, y_top,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=6,
                 fontweight='bold', color='#1a1a1a')
    for i, (bar, val) in enumerate(zip(bars_s, ap_sp_vals)):
        y_top = val + sd_sp_vals[i] + 0.02
        axC.text(bar.get_x() + bar.get_width()/2, y_top,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=6,
                 fontweight='bold', color=C_SP)

    axC.set_xticks(x)
    axC.set_xticklabels(methods, fontsize=7)
    axC.set_ylabel('Average Precision\n(ISFV as positive class)')
    axC.set_ylim([0, 1.20])
    axC.legend(loc='upper right', fontsize=6.5, ncol=2, columnspacing=1.0)
    axC.set_title('Evaluation Leakage Across Methods',
                  fontweight='bold', fontsize=9, pad=4)
    axC.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)

    # Panel labels (A, B, C)
    axA.text(-0.15, 1.12, 'A', transform=axA.transAxes,
             fontsize=14, fontweight='bold', va='top')
    axB.text(-0.15, 1.12, 'B', transform=axB.transAxes,
             fontsize=14, fontweight='bold', va='top')
    axC.text(-0.07, 1.10, 'C', transform=axC.transAxes,
             fontsize=14, fontweight='bold', va='top')

    for fmt in ('png', 'pdf'):
        fig.savefig(f'{OUT_MAIN}/Figure2.{fmt}', dpi=600)
    plt.close()
    print('Figure 2 saved.')


if __name__ == '__main__':
    make_figure2()
    for fmt in ('png', 'pdf'):
        src = f'{OUT_MAIN}/Figure2.{fmt}'
        dst = f'{OUT_REPO}/Figure2.{fmt}'
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(src, dst)
        print(f'  Copied Figure2.{fmt} -> reproducibility_repo/figures/')
    print('\nDone.')
