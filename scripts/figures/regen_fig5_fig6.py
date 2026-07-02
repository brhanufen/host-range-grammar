#!/usr/bin/env python3
"""
Regenerate Figure 5 (motif identification) and Figure 6 (robustness/ablation)
for the flavivirus manuscript.

Figure 5: unchanged content (3 panels: A=scatter, B=top 10 bars, C=UpA vs CpG deciles).
Figure 6B: adds Motif scorer as 4th bar alongside DNABERT-2, k-mer TF-IDF, Composition.
"""

import os, json, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ── Global publication style ─────────────────────────────────────────────
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

# Color palette
C_DNABERT = '#2171B5'
C_TFIDF  = '#E6550D'
C_COMP   = '#31A354'
C_POS    = '#969696'
C_MOTIF  = '#756BB1'
C_ACC    = '#2171B5'
C_SP     = '#E6550D'
C_CHANCE = '#BDBDBD'

# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_fig5_fig6.py -> repo root is two directories up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG  = os.path.dirname(REPO)

# Output directories
OUT_MAIN = os.path.join(REPO, 'figures')
OUT_REPO = os.path.join(REPO, 'figures')

# ── Load data ────────────────────────────────────────────────────────────
with open(os.path.join(REPO, 'results', 'all_baseline_results.json')) as f:
    res = json.load(f)
motifs = pd.read_csv(os.path.join(REPO, 'results', 'motif_enrichment_full.csv'))


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Discriminative 6-mer motif identification
# ══════════════════════════════════════════════════════════════════════════

def make_figure5():
    df = motifs.copy()

    df['category'] = 'Neither'
    df.loc[(df['n_UpA'] > 0) & (df['n_CpG'] == 0), 'category'] = 'UpA only'
    df.loc[(df['n_UpA'] == 0) & (df['n_CpG'] > 0), 'category'] = 'CpG only'
    df.loc[(df['n_UpA'] > 0) & (df['n_CpG'] > 0), 'category'] = 'UpA+CpG'

    # Colorblind-distinguishable palette + shape coding for robustness:
    # UpA-only = red circle, CpG-only = blue square, UpA+CpG = purple triangle, Neither = grey dot
    cat_colors = {
        'UpA only': '#D62728',
        'CpG only': '#2171B5',
        'UpA+CpG': '#6A3D9A',
        'Neither': '#BDBDBD',
    }
    cat_markers = {
        'UpA only': 'o',
        'CpG only': 's',
        'UpA+CpG': '^',
        'Neither': '.',
    }

    # Aspect ratio 1.67:1 (7.5×4.5") for vertical breathing room — panels were
    # cramped at 7.5×2.8 (2.68:1). Reviewer flagged Figure 5 as "stretched".
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 4.5),
                             gridspec_kw={'width_ratios': [1.1, 1, 1], 'wspace': 0.42})

    # ── Panel A: Volcano-style scatter ──
    ax = axes[0]
    # Draw dense red (UpA-only) points first, smaller and more transparent, so the
    # sparser blue (CpG-only) and purple (UpA+CpG) points remain visible on top
    # (reviewer comment: blue dots were covered by red).
    cat_style = {
        'Neither':  dict(size=3,  alpha=0.15, zorder=1, edge='none',  elw=0),
        'UpA only': dict(size=5,  alpha=0.35, zorder=2, edge='none',  elw=0),
        'UpA+CpG':  dict(size=14, alpha=0.85, zorder=4, edge='white', elw=0.3),
        'CpG only': dict(size=12, alpha=0.85, zorder=5, edge='white', elw=0.3),
    }
    for cat in ['Neither', 'UpA only', 'UpA+CpG', 'CpG only']:
        sub = df[df['category'] == cat]
        st = cat_style[cat]
        ax.scatter(sub['top_prev'], sub['log2_enrichment'], s=st['size'], alpha=st['alpha'],
                   c=cat_colors[cat], marker=cat_markers[cat],
                   label=cat, zorder=st['zorder'], edgecolors=st['edge'],
                   linewidths=st['elw'], rasterized=True)

    # Label top motifs — manually placed to avoid overlaps. Labels spread along
    # both axes so the leader arrows don't crowd into adjacent labels.
    # Convert DNA (T) to RNA (U) for display (reviewer comment: motifs shown as RNA).
    def dna_to_rna(s): return s.replace('T', 'U')
    top5 = df.nlargest(5, 'log2_enrichment')
    label_positions = {
        'ATTAGG': (0.20, 3.40),
        'CCCTTT': (0.55, 3.05),
        'AGAGAG': (0.78, 2.60),
        'TACTCC': (0.50, 2.05),
        'TAGTAT': (0.05, 2.30),
    }
    for _, row in top5.iterrows():
        motif_name = row['motif']
        if motif_name not in label_positions:
            continue
        tx, ty = label_positions[motif_name]
        ax.annotate(dna_to_rna(motif_name), xy=(row['top_prev'], row['log2_enrichment']),
                    xytext=(tx, ty), fontsize=5, fontweight='bold',
                    fontstyle='italic', zorder=5,
                    arrowprops=dict(arrowstyle='->', lw=0.5, color='#555555',
                                   shrinkA=0, shrinkB=2))

    ax.set_xlabel('Prevalence in top-5%\nARB predictions', fontsize=7)
    ax.set_ylabel('log$_2$(enrichment)')
    ax.axhline(0, color='gray', lw=0.5, ls='-', zorder=0)
    ax.legend(loc='lower right', markerscale=2.5, fontsize=5, handletextpad=0.3,
              borderpad=0.3)
    ax.set_title('6-mer Enrichment by\nDinucleotide Content', fontweight='bold',
                 fontsize=8, pad=4)

    # ── Panel B: Top 10 motifs horizontal bar ──
    ax = axes[1]
    top10 = df.nlargest(10, 'log2_enrichment').iloc[::-1]
    colors = [cat_colors[c] for c in top10['category']]
    bars = ax.barh(range(10), top10['log2_enrichment'], color=colors,
                   edgecolor='white', linewidth=0.5, height=0.65, zorder=3)

    ax.set_yticks(range(10))
    # Convert DNA → RNA for display (reviewer convention: motifs shown in RNA form)
    ax.set_yticklabels([m.replace('T', 'U') for m in top10['motif']],
                       fontfamily='monospace', fontsize=6.5, fontweight='bold')
    ax.set_xlabel('log$_2$(enrichment)')
    ax.set_title('Top 10 ARB-Enriched\nMotifs', fontweight='bold', fontsize=8, pad=4)

    # UpA count labels inside bars
    for i, (_, row) in enumerate(top10.iterrows()):
        if row['n_UpA'] > 0:
            ax.text(row['log2_enrichment'] - 0.08, i,
                    f"UpA×{int(row['n_UpA'])}",
                    va='center', ha='right', fontsize=4.5, color='white',
                    fontweight='bold')

    ax.xaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)
    ax.set_xlim([0, 3.5])

    # ── Panel C: UpA vs CpG by enrichment decile ──
    ax = axes[2]
    df_sorted = df.sort_values('log2_enrichment', ascending=False).reset_index(drop=True)
    df_sorted['decile'] = pd.qcut(range(len(df_sorted)), 10, labels=range(1, 11))

    deciles = range(1, 11)
    upa_frac = []
    cpg_frac = []
    for d in deciles:
        sub = df_sorted[df_sorted['decile'] == d]
        # "UpA-only" / "CpG-only" (exclude UpA+CpG) to match the caption and the
        # in-text fractions (24% expected; ARB-decile UpA-only 37%, ISFV-decile CpG-only 66%).
        upa_frac.append((sub['category'] == 'UpA only').mean())
        cpg_frac.append((sub['category'] == 'CpG only').mean())

    x = np.arange(1, 11)
    w = 0.35
    ax.bar(x - w/2, upa_frac, w, color='#D62728', edgecolor='white', linewidth=0.5,
           label='UpA-only', zorder=3)
    ax.bar(x + w/2, cpg_frac, w, color='#2171B5', edgecolor='white', linewidth=0.5,
           label='CpG-only', zorder=3)

    ax.set_xlabel('Enrichment decile\n(1 = most ARB-enriched)', fontsize=7)
    ax.set_ylabel('Fraction of motifs')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    ax.legend(loc='upper left', fontsize=5.5)
    ax.set_title('UpA vs CpG by\nEnrichment Rank', fontweight='bold',
                 fontsize=8, pad=4)
    ax.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)

    # Panel labels
    for i, letter in enumerate('ABC'):
        axes[i].text(-0.16, 1.18, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight='bold', va='top')

    for fmt in ('png', 'pdf'):
        fig.savefig(f'{OUT_MAIN}/Figure5.{fmt}', dpi=600)
    plt.close()
    print('Figure 5 saved.')


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Robustness and ablation analyses (with Motif scorer added)
# ══════════════════════════════════════════════════════════════════════════

def make_figure6():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.8),
                             gridspec_kw={'wspace': 0.4})

    # ── Panel A: Window masking ablation (REPRODUCIBLE — from ablation_results_repro.json) ──
    ax = axes[0]
    import json as _json
    _abl = _json.load(open(
        os.path.join(REPO, 'results', 'ablation_results_repro.json')
    ))['conditions']
    conditions = ['Baseline\n(no masking)', 'Top-5%\nper genome',
                  'Random-5%\nper genome', 'AUUAGG-\ncontaining']
    keys = ['baseline', 'top5_per_genome', 'random5_per_genome', 'auuagg_containing']
    ap_vals = [_abl[k]['AP_mean'] for k in keys]
    ap_stds = [_abl[k]['AP_std'] for k in keys]
    colors_abl = [C_DNABERT, C_TFIDF, C_COMP, C_MOTIF]

    bars = ax.bar(range(4), ap_vals, yerr=ap_stds, color=colors_abl, edgecolor='white',
                  linewidth=0.5, width=0.6, zorder=3,
                  error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8})

    # Value labels above bars (4 decimals to make the tiny differences visible)
    for bar, val, sd in zip(bars, ap_vals, ap_stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sd + 0.015,
                f'{val:.4f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    # Δ annotations (signed, in pp) — one per non-baseline bar, positioned above the value label
    for i in [1, 2, 3]:
        delta_pp = (ap_vals[i] - ap_vals[0]) * 100
        if abs(delta_pp) < 0.001:
            delta_text = 'Δ ≈ 0 pp'
        else:
            sign = '+' if delta_pp >= 0 else '−'
            delta_text = f'Δ = {sign}{abs(delta_pp):.3f} pp'
        # Place above the value label (dark text) so it is always legible
        ax.text(i, ap_vals[i] + ap_stds[i] + 0.058, delta_text,
                fontsize=5.5, color='#444444', ha='center', va='bottom',
                fontweight='bold', rotation=0)

    ax.set_xticks(range(4))
    ax.set_xticklabels(conditions, fontsize=5.5)
    ax.set_ylabel('Average Precision')
    ax.set_ylim([0, 1.18])
    ax.set_title('Window Masking Ablation\n(all |Δ| < 0.01 pp)', fontweight='bold', fontsize=8, pad=6)
    ax.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)

    # ── Panel B: Method comparison — now with 4 methods including Motif scorer ──
    ax = axes[1]
    methods = ['DNABERT-2', 'k-mer\nTF-IDF', 'Compo-\nsition', 'Motif\nscorer']
    method_keys = ['DNABERT-2', 'k-mer TF-IDF', 'Composition', 'Motif scorer (top 50)']

    ap_acc_vals = [res[k]['accession']['ap_mean'] for k in method_keys]
    ap_sp_vals  = [res[k]['species']['ap_mean']   for k in method_keys]
    sd_acc_vals = [res[k]['accession']['ap_std']   for k in method_keys]
    sd_sp_vals  = [res[k]['species']['ap_std']     for k in method_keys]

    x = np.arange(len(methods))
    w = 0.32
    bars_a = ax.bar(x - w/2, ap_acc_vals, w, yerr=sd_acc_vals, color=C_ACC,
                     edgecolor='white', linewidth=0.5,
                     error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8},
                     label='Accession-grouped', zorder=3)
    bars_s = ax.bar(x + w/2, ap_sp_vals, w, yerr=sd_sp_vals, color=C_SP,
                     edgecolor='white', linewidth=0.5,
                     error_kw={'lw': 0.8, 'capsize': 2, 'capthick': 0.8},
                     label='Species-grouped', zorder=3)

    # Value labels — ALWAYS placed above the error bar whisker to avoid overlap.
    # Both accession and species labels use the same convention.
    for i, (bar, val) in enumerate(zip(bars_a, ap_acc_vals)):
        y_top = val + sd_acc_vals[i] + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, y_top,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5.5,
                fontweight='bold', color='#1a1a1a')
    for i, (bar, val) in enumerate(zip(bars_s, ap_sp_vals)):
        y_top = val + sd_sp_vals[i] + 0.02
        ax.text(bar.get_x() + bar.get_width()/2, y_top,
                f'{val:.3f}', ha='center', va='bottom', fontsize=5.5,
                fontweight='bold', color=C_SP)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=6.5)
    ax.set_ylabel('Average Precision')
    ax.set_ylim([0, 1.18])
    ax.legend(loc='upper center', fontsize=5.5, ncol=2, columnspacing=1.0,
              bbox_to_anchor=(0.5, 1.0))
    ax.set_title('Method Comparison\nAcross Evaluation Schemes', fontweight='bold',
                 fontsize=8, pad=4)
    ax.yaxis.grid(True, alpha=0.3, lw=0.5, zorder=0)

    # Panel labels
    for i, letter in enumerate('AB'):
        axes[i].text(-0.16, 1.12, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight='bold', va='top')

    for fmt in ('png', 'pdf'):
        fig.savefig(f'{OUT_MAIN}/Figure6.{fmt}', dpi=600)
    plt.close()
    print('Figure 6 saved.')


# ── Run & copy ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    make_figure5()
    make_figure6()

    # Copy to reproducibility repo
    for fig_name in ('Figure5', 'Figure6'):
        for fmt in ('png', 'pdf'):
            src = f'{OUT_MAIN}/{fig_name}.{fmt}'
            dst = f'{OUT_REPO}/{fig_name}.{fmt}'
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            print(f'  Copied {fig_name}.{fmt} -> reproducibility_repo/figures/')

    print('\nDone. Figures 5 and 6 regenerated at 600 DPI.')
