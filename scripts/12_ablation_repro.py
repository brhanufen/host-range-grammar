#!/usr/bin/env python3
"""
Fix 2 — Reproducible window-masking ablation for P85 / Figure 6A.

For each condition (baseline / top-5% per genome / random-5% per genome /
AUUAGG-positive windows), recompute genome-level embeddings as the mean of
unmasked window embeddings, then evaluate a logistic regression with
ARB-vs-ISV labels using accession-grouped 5-fold CV (SEED=42).

We use **genome-grouped** CV here (each genome appears once at the genome
level), keeping all other parameters identical to the main classifier
methodology: balanced LR, max_iter=1000, random_state=42.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold

SEED = 42
N_SPLITS = 5
# Portable paths: resolve repo root from this file's location (scripts/ -> repo root)
REPO = Path(__file__).resolve().parent.parent
EMBEDDINGS_PATH = REPO / "embeddings" / "embeddings_combined.npy"
WINDOWS_META = REPO / "data" / "windows_metadata.csv"
WINDOWS_SEQ = REPO / "data" / "windows_sequences.csv"
OUTPUT = REPO / "results" / "ablation_results_repro.json"

print("Loading embeddings, windows metadata, and sequences...")
emb = np.load(EMBEDDINGS_PATH).astype(np.float32)
wm = pd.read_csv(WINDOWS_META)
ws = pd.read_csv(WINDOWS_SEQ)

# Align sequences with metadata
wm = wm.merge(ws, on="window_id", how="left")
assert wm["sequence"].notna().all(), "Some windows missing sequence"
print(f"  {len(wm)} windows, {wm['accession'].nunique()} genomes")

# Label: ARB = 1 (positive), ISV = 0 (matches main analysis)
# Note: oof_predictions.npz uses y_isv (ISV=1, ARB=0); we'll keep ARB=1 here
# for genome-level eval, which is symmetric for AP/AUC interpretation.
wm["y_arb"] = (wm["label"] == "ARB").astype(int)

def compute_genome_embeddings(mask: np.ndarray):
    """Average window embeddings per genome, excluding masked windows.
    Returns (X, y, genomes, species) at the genome level.
    """
    X_list, y_list, genome_list, species_list = [], [], [], []
    for acc, sub in wm.groupby("accession", sort=False):
        idx = sub.index.values
        keep = idx[~mask[idx]]
        if len(keep) == 0:
            keep = idx  # fallback if everything masked
        g_emb = emb[keep].mean(axis=0)
        X_list.append(g_emb)
        y_list.append(int(sub["y_arb"].iloc[0]))
        genome_list.append(acc)
        species_list.append(sub["species"].iloc[0])
    return np.vstack(X_list), np.array(y_list), np.array(genome_list), np.array(species_list)

def eval_genome_cv(X, y, groups):
    """Species-grouped k-fold at genome level.
    Matches original 09_ablation_studies.py methodology.
    Returns mean ± std AP across folds.
    """
    gkf = GroupKFold(n_splits=N_SPLITS)
    aps, aucs = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], p))
        aucs.append(roc_auc_score(y[te], p))
    return {
        "AP_mean": float(np.mean(aps)),
        "AP_std": float(np.std(aps)),
        "AUC_mean": float(np.mean(aucs)),
        "AUC_std": float(np.std(aucs)),
        "AP_per_fold": [float(a) for a in aps],
    }

# --- Baseline: no masking ---
print("\nBaseline (no masking)...")
mask_base = np.zeros(len(wm), dtype=bool)
X, y, _, sp = compute_genome_embeddings(mask_base)
print(f"  Genome embeddings: {X.shape}, ARB={int(y.sum())} ISV={int((1-y).sum())}")
baseline = eval_genome_cv(X, y, sp)
print(f"  Baseline AP = {baseline['AP_mean']:.4f} ± {baseline['AP_std']:.4f}")

# --- Ablation 1: Mask top 5% windows per genome (by reproducible P(ARB)) ---
# Use the reproducible OOF P(ARB) we previously computed
print("\nAblation 1: Mask top 5% windows per genome (by P(ARB))...")
# Reproducible OOF P(ARB) ships in data/windows_predictions_FIXED_SEED.csv;
# merge on window_id so row order is guaranteed to align with wm.
_pred = pd.read_csv(REPO / "data" / "windows_predictions_FIXED_SEED.csv",
                    usecols=["window_id", "p_arb"])
wm = wm.merge(_pred, on="window_id", how="left")
assert wm["p_arb"].notna().all(), "P(ARB) merge left unmatched windows"

mask_top5 = np.zeros(len(wm), dtype=bool)
for acc, sub in wm.groupby("accession", sort=False):
    thresh = sub["p_arb"].quantile(0.95)
    mask_top5[sub.index[sub["p_arb"] >= thresh]] = True
print(f"  Masked: {mask_top5.sum()} ({100*mask_top5.mean():.1f}%)")
X, y, _, sp = compute_genome_embeddings(mask_top5)
top5 = eval_genome_cv(X, y, sp)
print(f"  Top-5% AP = {top5['AP_mean']:.4f} ± {top5['AP_std']:.4f}   ΔAP = {(top5['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")

# --- Ablation 2: Mask random 5% per genome ---
print("\nAblation 2: Mask random 5% windows per genome...")
rng = np.random.default_rng(SEED)
mask_rand5 = np.zeros(len(wm), dtype=bool)
for acc, sub in wm.groupby("accession", sort=False):
    idx = sub.index.values
    n_mask = max(1, int(len(idx) * 0.05))
    chosen = rng.choice(idx, size=n_mask, replace=False)
    mask_rand5[chosen] = True
print(f"  Masked: {mask_rand5.sum()} ({100*mask_rand5.mean():.1f}%)")
X, y, _, sp = compute_genome_embeddings(mask_rand5)
rand5 = eval_genome_cv(X, y, sp)
print(f"  Rand-5% AP = {rand5['AP_mean']:.4f} ± {rand5['AP_std']:.4f}   ΔAP = {(rand5['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")

# --- Ablation 3: Mask all windows containing ATTAGG (AUUAGG in RNA) ---
print("\nAblation 3: Mask all windows containing ATTAGG (AUUAGG)...")
mask_motif = wm["sequence"].str.contains("ATTAGG", regex=False, na=False).values
print(f"  Masked: {mask_motif.sum()} ({100*mask_motif.mean():.1f}%)")
X, y, _, sp = compute_genome_embeddings(mask_motif)
motif = eval_genome_cv(X, y, sp)
print(f"  AUUAGG-masked AP = {motif['AP_mean']:.4f} ± {motif['AP_std']:.4f}   ΔAP = {(motif['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")

# --- Save ---
results = {
    "methodology": "Genome-level mean-pool of DNABERT-2 window embeddings, excluding masked windows; "
                   "balanced LR, stratified 5-fold CV at genome level, SEED=42",
    "n_genomes": int(wm["accession"].nunique()),
    "n_windows": int(len(wm)),
    "conditions": {
        "baseline": {"AP_mean": baseline["AP_mean"], "AP_std": baseline["AP_std"], "delta_AP_pp": 0.0, "n_masked": 0},
        "top5_per_genome": {"AP_mean": top5["AP_mean"], "AP_std": top5["AP_std"], "delta_AP_pp": (top5["AP_mean"]-baseline["AP_mean"])*100, "n_masked": int(mask_top5.sum())},
        "random5_per_genome": {"AP_mean": rand5["AP_mean"], "AP_std": rand5["AP_std"], "delta_AP_pp": (rand5["AP_mean"]-baseline["AP_mean"])*100, "n_masked": int(mask_rand5.sum())},
        "auuagg_containing": {"AP_mean": motif["AP_mean"], "AP_std": motif["AP_std"], "delta_AP_pp": (motif["AP_mean"]-baseline["AP_mean"])*100, "n_masked": int(mask_motif.sum())},
    },
}

Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT}")

print("\n" + "="*70)
print("SUMMARY (REPRODUCIBLE)")
print("="*70)
print(f"  Baseline:           AP = {baseline['AP_mean']:.4f} ± {baseline['AP_std']:.4f}")
print(f"  Top 5% masking:     AP = {top5['AP_mean']:.4f} ± {top5['AP_std']:.4f}   Δ = {(top5['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")
print(f"  Random 5% masking:  AP = {rand5['AP_mean']:.4f} ± {rand5['AP_std']:.4f}   Δ = {(rand5['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")
print(f"  AUUAGG masking:     AP = {motif['AP_mean']:.4f} ± {motif['AP_std']:.4f}   Δ = {(motif['AP_mean']-baseline['AP_mean'])*100:+.2f} pp")
print("="*70)
