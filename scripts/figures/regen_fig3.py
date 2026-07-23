#!/usr/bin/env python3
"""Regenerate Figure 3 (Embedding space organization) reproducibly.

Panels:
  A: UMAP projection of genome-mean DNABERT-2 embeddings, colored by host-range class
  B: t-SNE projection, same data
  C: Clustering metrics — kNN purity (0.997) vs permutation null + silhouette score

All values reproducible from windows_predictions_FIXED_SEED.csv methodology:
  SEED=42, np.random.seed, sklearn random_state=42
"""
import warnings; warnings.filterwarnings('ignore')
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import umap

rcParams.update({
    'font.family': 'Arial', 'font.size': 8,
    'figure.dpi': 600, 'savefig.dpi': 600, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

SEED = 42
np.random.seed(SEED)
import random
random.seed(SEED)

import os
# Paths resolved relative to this script so the repo is portable.
# scripts/figures/regen_fig3.py -> repo root is two directories up.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG  = os.path.dirname(REPO)
OUT_MAIN = os.path.join(REPO, 'figures')
OUT_REPO = f'{REPO}/figures'
EMBEDDINGS = os.path.join(REPO, 'embeddings', 'embeddings_combined.npy')

C_ARB = '#2171B5'
C_ISV = '#E6550D'

# ───────── Load data ─────────
print("Loading embeddings and metadata...")
emb = np.load(EMBEDDINGS).astype(np.float32)
wm = pd.read_csv(f"{REPO}/data/windows_metadata.csv")
print(f"  {emb.shape[0]} windows × {emb.shape[1]} dim")
print(f"  {wm['accession'].nunique()} genomes")

# ───────── Compute genome-level embeddings ─────────
print("\nComputing genome-mean embeddings...")
G_emb, G_lab = [], []
for acc, sub in wm.groupby('accession', sort=False):
    idx = sub.index.values
    G_emb.append(emb[idx].mean(axis=0))
    G_lab.append(1 if sub['label'].iloc[0] == 'ISV' else 0)
G_emb = np.vstack(G_emb).astype(np.float32)
G_lab = np.array(G_lab)
print(f"  Genome embeddings: {G_emb.shape}")
print(f"  ARB={int((G_lab==0).sum())}, ISV={int((G_lab==1).sum())}")

# ───────── UMAP ─────────
print("\nRunning UMAP...")
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                       random_state=SEED, n_jobs=1)
umap_xy = umap_model.fit_transform(G_emb)
print(f"  UMAP range: x=[{umap_xy[:,0].min():.1f}, {umap_xy[:,0].max():.1f}]  "
      f"y=[{umap_xy[:,1].min():.1f}, {umap_xy[:,1].max():.1f}]")

# ───────── t-SNE ─────────
print("\nRunning t-SNE...")
tsne_model = TSNE(n_components=2, perplexity=30, random_state=SEED,
                  init='pca', learning_rate='auto')
tsne_xy = tsne_model.fit_transform(G_emb)
print(f"  t-SNE range: x=[{tsne_xy[:,0].min():.1f}, {tsne_xy[:,0].max():.1f}]  "
      f"y=[{tsne_xy[:,1].min():.1f}, {tsne_xy[:,1].max():.1f}]")

# ───────── kNN purity & permutation null ─────────
print("\nComputing kNN purity (k=15)...")
k = 15
nn = NearestNeighbors(n_neighbors=k + 1).fit(G_emb)
_, nn_idx = nn.kneighbors(G_emb)
nn_idx = nn_idx[:, 1:]  # exclude self
neighbor_labels = G_lab[nn_idx]
own = G_lab[:, None]
purity = (neighbor_labels == own).mean(axis=1).mean()
print(f"  kNN purity = {purity:.4f}")

print("Computing permutation null (1000 perms)...")
rng = np.random.default_rng(SEED)
null_purities = []
for _ in range(1000):
    shuffled = rng.permutation(G_lab)
    nl = shuffled[nn_idx]
    own_sh = shuffled[:, None]
    null_purities.append((nl == own_sh).mean(axis=1).mean())
null_mean = float(np.mean(null_purities))
null_std = float(np.std(null_purities))
z_score = (purity - null_mean) / null_std
print(f"  Null: {null_mean:.4f} ± {null_std:.4f}, Z = {z_score:.1f}")

# Silhouette
sil = float(silhouette_score(G_emb, G_lab))
print(f"  Silhouette = {sil:.4f}")

# ───────── Build figure ─────────
print("\nBuilding figure...")
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5),
                         gridspec_kw={'width_ratios': [1, 1, 0.6], 'wspace': 0.35})

# Panel A: UMAP
ax = axes[0]
mask_arb = G_lab == 0
mask_isv = G_lab == 1
ax.scatter(umap_xy[mask_arb, 0], umap_xy[mask_arb, 1], s=6, c=C_ARB, alpha=0.6,
           edgecolors='none', label=f'ARB (n={int(mask_arb.sum())})', rasterized=True)
ax.scatter(umap_xy[mask_isv, 0], umap_xy[mask_isv, 1], s=10, c=C_ISV, alpha=0.85,
           edgecolors='none', label=f'ISFV (n={int(mask_isv.sum())})', rasterized=True)
ax.set_xlabel('UMAP 1', fontsize=9)
ax.set_ylabel('UMAP 2', fontsize=9)
_leg_handles, _leg_labels = ax.get_legend_handles_labels()
ax.text(-0.18, 1.05, 'A', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel B: t-SNE
ax = axes[1]
ax.scatter(tsne_xy[mask_arb, 0], tsne_xy[mask_arb, 1], s=6, c=C_ARB, alpha=0.6,
           edgecolors='none', label=f'ARB (n={int(mask_arb.sum())})', rasterized=True)
ax.scatter(tsne_xy[mask_isv, 0], tsne_xy[mask_isv, 1], s=10, c=C_ISV, alpha=0.85,
           edgecolors='none', label=f'ISFV (n={int(mask_isv.sum())})', rasterized=True)
ax.set_xlabel('t-SNE 1', fontsize=9)
ax.set_ylabel('t-SNE 2', fontsize=9)
ax.text(-0.18, 1.05, 'B', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# Panel C: kNN purity + silhouette
ax = axes[2]
bars = ax.bar(['kNN Purity\n(k=15)', 'Silhouette\nScore'], [purity, sil],
              color=[C_ARB, '#777777'], edgecolor='white', linewidth=0.5, width=0.55)
# Null reference line
ax.axhline(null_mean, color='#999999', ls='--', lw=1.0, label=f'Null\n{null_mean:.3f}±{null_std:.3f}')
# Value labels above bars
for bar, val in zip(bars, [purity, sil]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
# Z-score and p annotation (anchored from the top so the stars sit a little
# below the top axis edge rather than crammed against it)
ax.text(0, 1.15, '***\nP < 0.001',
        ha='center', va='top', fontsize=7, color='black')
ax.set_ylabel('Score', fontsize=9)
ax.set_ylim([0, 1.20])
ax.legend(loc='center right', fontsize=6.5, frameon=True)
ax.text(-0.30, 1.05, 'C', transform=ax.transAxes, fontsize=14, fontweight='bold', va='top')

# make room at the bottom, THEN place a shared ARB/ISFV legend under panels A & B
fig.subplots_adjust(bottom=0.22)
posA = axes[0].get_position(); posB = axes[1].get_position()
_mid_x = (posA.x0 + posB.x1) / 2.0
fig.legend(_leg_handles, _leg_labels, loc='upper center',
           bbox_to_anchor=(_mid_x, posA.y0 - 0.03), ncol=2, fontsize=8,
           markerscale=2, frameon=False, handletextpad=0.4, columnspacing=1.6)

for fmt in ('png', 'pdf'):
    out = f'{OUT_MAIN}/Figure3.{fmt}'
    fig.savefig(out, dpi=600)
    print(f'  Saved: {out}')
    _dst = f'{OUT_REPO}/Figure3.{fmt}'
    if os.path.abspath(out) != os.path.abspath(_dst):
        shutil.copy2(out, _dst)

plt.close()
print(f'\nReproducible values:')
print(f'  kNN purity:   {purity:.4f} (target: 0.997)')
print(f'  Null mean:    {null_mean:.4f} (target: 0.868)')
print(f'  Z-score:      {z_score:.1f} (target: 43.9)')
print(f'  Silhouette:   {sil:.4f} (target: 0.433)')
print('\nDone.')
