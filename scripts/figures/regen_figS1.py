#!/usr/bin/env python3
"""
Regenerate Supplementary Figure S1 (composition-feature separation) and
Supplementary Table S2 for the flavivirus manuscript.

Supplementary Figure S1 characterizes which of the 17 composition features
(GC content + 16 dinucleotide frequencies) separate arbovirus (ARB) from
insect-specific flavivirus (ISFV) windows:
  (A) per-feature effect size (Cohen's d), coloured by enriched class
  (B) CpG dinucleotide frequency distribution by class (violin + inset box)
  (C) UpA dinucleotide frequency distribution by class

Supplementary Table S2 tabulates, for all 17 features: mean by class,
Cohen's d (ISFV - ARB), AUC, and Mann-Whitney U P value.

Feature definitions match the composition-only baseline in
scripts/10_baseline_comparison.py (GC content + 16 dinucleotide frequencies,
overlapping dinucleotide counts).

Usage:
    cd scripts/figures && python regen_figS1.py
Outputs (written to <repo>/supplementary/):
    FigureS1.pdf, FigureS1.png, Supplementary_Table_S2.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

# -- Global publication style -------------------------------------------------
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

C_ARB = '#2171B5'
C_ISV = '#E6550D'

# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_figS1.py -> repo root is two directories up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, 'supplementary')
os.makedirs(OUT, exist_ok=True)


def compute_composition_features(sequences):
    """GC content + 16 dinucleotide frequencies (overlapping counts).

    Matches compute_dinucleotide_features() in scripts/10_baseline_comparison.py.
    """
    dinucs = [a + b for a in 'ACGT' for b in 'ACGT']
    feats = np.empty((len(sequences), 17), dtype=np.float64)
    for r, seq in enumerate(sequences):
        seq = seq.upper()
        n = len(seq)
        gc = (seq.count('G') + seq.count('C')) / n if n > 0 else 0.0
        row = [gc]
        for di in dinucs:
            count = sum(1 for i in range(n - 1) if seq[i:i + 2] == di)
            row.append(count / (n - 1) if n > 1 else 0.0)
        feats[r] = row
    return feats, ['GC_content'] + dinucs


def main():
    print("=" * 70)
    print("REGENERATE SUPPLEMENTARY FIGURE S1 + TABLE S2")
    print("=" * 70)

    # -- Load data --
    print("\n[1/4] Loading windows...")
    wm = pd.read_csv(os.path.join(REPO, 'data', 'windows_metadata.csv'))
    seq = pd.read_csv(os.path.join(REPO, 'data', 'windows_sequences.csv'))
    df = seq.merge(wm[['window_id', 'label']], on='window_id', how='inner')
    print(f"  Windows: {len(df)}")
    # label is 'ARB' / 'ISV' in the data; display as ISFV
    n_arb = int((df['label'] == 'ARB').sum())
    n_isv = int((df['label'] != 'ARB').sum())
    print(f"  ARB: {n_arb}   ISFV: {n_isv}")

    # -- Composition features --
    print("\n[2/4] Computing 17 composition features (GC + 16 dinucleotides)...")
    feats, cols = compute_composition_features(df['sequence'].values)
    fdf = pd.DataFrame(feats, columns=cols)
    fdf['label'] = df['label'].values
    arb_mask = (fdf['label'] == 'ARB').values
    isv_mask = ~arb_mask

    # -- Per-feature separation stats --
    print("\n[3/4] Per-feature separation (Cohen's d, AUC, Mann-Whitney)...")
    rows = []
    for c in cols:
        x = fdf[c].values
        a = x[arb_mask]
        i = x[isv_mask]
        U, p = mannwhitneyu(i, a, alternative='two-sided')  # ISFV vs ARB
        auc = U / (n_arb * n_isv)                            # AUC = P(ISFV > ARB)
        sp = np.sqrt(((n_arb - 1) * a.var(ddof=1) +
                      (n_isv - 1) * i.var(ddof=1)) / (n_arb + n_isv - 2))
        d = (i.mean() - a.mean()) / sp if sp > 0 else 0.0
        rows.append((c, a.mean(), i.mean(), d, auc, p))
    res = pd.DataFrame(rows, columns=['feature', 'mean_ARB', 'mean_ISV',
                                      'cohend', 'auc', 'p'])
    res['abs_d'] = res['cohend'].abs()
    cpg = res.loc[res.feature == 'CG'].iloc[0]
    upa = res.loc[res.feature == 'TA'].iloc[0]
    print(f"  CpG: ARB {cpg.mean_ARB:.4f}  ISFV {cpg.mean_ISV:.4f}  "
          f"d={cpg.cohend:.2f}  AUC={cpg.auc:.3f}")
    print(f"  UpA: d={upa.cohend:.2f}  AUC={upa.auc:.3f}")

    # -- Figure --
    print("\n[4/4] Rendering Supplementary Figure S1...")
    alias = {'CG': 'CpG', 'TA': 'UpA', 'GC_content': 'GC content'}
    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.42)

    # Panel A: signed effect size, coloured by enriched class
    axA = fig.add_subplot(gs[0, 0])
    rs = res.sort_values('cohend')  # ascending for horizontal bars
    labels = [alias.get(f, f) for f in rs['feature']]
    colors = [C_ISV if v > 0 else C_ARB for v in rs['cohend']]
    yy = np.arange(len(rs))
    axA.barh(yy, rs['cohend'], color=colors, edgecolor='white',
             linewidth=0.4, zorder=3)
    axA.set_yticks(yy)
    axA.set_yticklabels(labels, fontsize=5.5)
    axA.axvline(0, color='#333333', lw=0.6, zorder=2)
    axA.set_xlabel("Effect size (Cohen's d)\nISFV higher \u2192", fontsize=7)
    axA.set_title("Composition-feature\nseparation (ARB vs ISFV)", fontsize=7.5)
    axA.tick_params(axis='x', labelsize=6)
    axA.legend(handles=[Patch(fc=C_ARB, label='ARB-enriched'),
                        Patch(fc=C_ISV, label='ISFV-enriched')],
               loc='lower right', fontsize=5.2, frameon=False,
               handlelength=1.0, handletextpad=0.4)
    axA.grid(axis='x', alpha=0.3, lw=0.5, zorder=0)

    def half_violins(ax, colname):
        a = fdf.loc[fdf.label == 'ARB', colname].values
        i = fdf.loc[fdf.label != 'ARB', colname].values
        parts = ax.violinplot([a, i], positions=[0, 1],
                              showextrema=False, widths=0.8)
        for pc, col in zip(parts['bodies'], [C_ARB, C_ISV]):
            pc.set_facecolor(col); pc.set_alpha(0.55)
            pc.set_edgecolor(col); pc.set_linewidth(0.6)
        bp = ax.boxplot([a, i], positions=[0, 1], widths=0.14,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color='white', lw=1.0), zorder=5)
        for patch, col in zip(bp['boxes'], [C_ARB, C_ISV]):
            patch.set_facecolor(col); patch.set_edgecolor('none')
        for w in bp['whiskers']:
            w.set_color('#555555'); w.set_linewidth(0.7)
        for cap in bp['caps']:
            cap.set_color('#555555'); cap.set_linewidth(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f'ARB\n(n={n_arb})', f'ISFV\n(n={n_isv})'], fontsize=6)
        ax.grid(axis='y', alpha=0.3, lw=0.5, zorder=0)

    # Panel B: CpG
    axB = fig.add_subplot(gs[0, 1])
    half_violins(axB, 'CG')
    axB.set_title('CpG frequency', fontsize=7.5)
    axB.set_ylabel('Dinucleotide frequency', fontsize=7)
    axB.text(0.5, 0.97, f"AUC = {cpg.auc:.2f}", transform=axB.transAxes,
             ha='center', va='top', fontsize=6, color='#333333')

    # Panel C: UpA
    axC = fig.add_subplot(gs[0, 2])
    half_violins(axC, 'TA')
    axC.set_title('UpA frequency', fontsize=7.5)
    axC.text(0.5, 0.97, f"AUC = {upa.auc:.2f}", transform=axC.transAxes,
             ha='center', va='top', fontsize=6, color='#333333')

    # Panel letters
    for ax, L in [(axA, 'a'), (axB, 'b'), (axC, 'c')]:
        ax.text(-0.18, 1.08, L, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='top', ha='right')

    fig.savefig(os.path.join(OUT, 'FigureS1.pdf'))
    fig.savefig(os.path.join(OUT, 'FigureS1.png'), dpi=600)
    plt.close(fig)

    # -- Table S2 --
    tab = res.sort_values('abs_d', ascending=False)[
        ['feature', 'mean_ARB', 'mean_ISV', 'cohend', 'auc', 'p']].copy()
    tab['feature'] = tab['feature'].map(lambda f: alias.get(f, f))
    tab = tab.rename(columns={
        'feature': 'Feature', 'mean_ARB': 'Mean (ARB)',
        'mean_ISV': 'Mean (ISFV)', 'cohend': "Cohen's d (ISFV-ARB)",
        'auc': 'AUC', 'p': 'P (Mann-Whitney)'})
    tab['P (Mann-Whitney)'] = tab['P (Mann-Whitney)'].map(lambda v: f"{v:.2e}")
    for c in ['Mean (ARB)', 'Mean (ISFV)']:
        tab[c] = tab[c].map(lambda v: f"{v:.4f}")
    tab["Cohen's d (ISFV-ARB)"] = tab["Cohen's d (ISFV-ARB)"].map(lambda v: f"{v:+.3f}")
    tab['AUC'] = tab['AUC'].map(lambda v: f"{v:.3f}")
    tab.to_csv(os.path.join(OUT, 'Supplementary_Table_S2.csv'), index=False)

    print("\nDone. Wrote FigureS1.pdf/png and Supplementary_Table_S2.csv to "
          + os.path.relpath(OUT, REPO) + "/")


if __name__ == '__main__':
    main()
