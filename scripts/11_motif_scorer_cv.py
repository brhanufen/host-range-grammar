#!/usr/bin/env python3
"""
11_motif_scorer_cv.py
=====================
Honest evaluation of the transparent motif-based scorer under both
accession-grouped and species-grouped cross-validation.

Replaces the leakage-contaminated 0.992 vs 0.697 comparison in
manuscript draft 7 (P85-P86) with reproducible numbers.

Two scoring approaches are evaluated:
  (a) Transparent score: sum of (motif presence x log2 enrichment) per window,
      used directly as the discriminative score. This matches the original
      manuscript description.
  (b) Logistic regression on top-50 motif presence features. This is the
      methodologically standard analog of the other baselines (TF-IDF,
      composition).

Both are run under accession-grouped and species-grouped 5-fold CV using
identical groupings to 05_grouped_cv.py.

Usage:
    python scripts/11_motif_scorer_cv.py \
        --sequences data/windows_sequences.csv \
        --windows data/windows_metadata.csv \
        --motifs results/motif_enrichment_full.csv \
        --output results/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold


def build_motif_features(sequences: pd.Series, motifs: list) -> np.ndarray:
    """
    Build a binary presence matrix for top motifs across windows.
    Each cell is 1 if the motif appears at least once in the window sequence.
    """
    n_windows = len(sequences)
    n_motifs = len(motifs)
    features = np.zeros((n_windows, n_motifs), dtype=np.int8)
    for j, motif in enumerate(motifs):
        for i, seq in enumerate(sequences):
            if motif in seq:
                features[i, j] = 1
    return features


def transparent_score(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Sum of (motif presence x log2 enrichment) per window.
    Higher score = more ARB-discriminative (since enriched motifs are
    overrepresented in high-p_arb windows).
    """
    return features @ weights


def evaluate_score(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5, invert_for_isv: bool = True) -> dict:
    """
    Compute per-fold AP and AUC for a fixed per-window score under GroupKFold.
    The score is used directly as the ranking (no per-fold retraining).

    invert_for_isv: y=1 means ISV. Motif score is ARB-discriminative,
    so we use -score to predict ISV.
    """
    if invert_for_isv:
        pred = -scores
    else:
        pred = scores

    gkf = GroupKFold(n_splits=n_splits)
    aps, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(scores, y, groups)):
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        ap = average_precision_score(y_test, pred[test_idx])
        auc = roc_auc_score(y_test, pred[test_idx])
        aps.append(ap)
        aucs.append(auc)
    return {
        'ap_mean': float(np.mean(aps)),
        'ap_std': float(np.std(aps)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'aps': [float(x) for x in aps],
        'aucs': [float(x) for x in aucs],
    }


def evaluate_logreg(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = 5) -> dict:
    """
    Run logistic regression on motif presence features under GroupKFold.
    Mirrors the protocol in 05_grouped_cv.py exactly.
    """
    gkf = GroupKFold(n_splits=n_splits)
    aps, aucs = [], []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = LogisticRegression(max_iter=1000, class_weight='balanced',
                                 random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        aps.append(float(average_precision_score(y_test, proba)))
        aucs.append(float(roc_auc_score(y_test, proba)))
    return {
        'ap_mean': float(np.mean(aps)),
        'ap_std': float(np.std(aps)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'aps': aps,
        'aucs': aucs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', required=True)
    parser.add_argument('--windows', required=True)
    parser.add_argument('--motifs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--n-top-motifs', type=int, default=50)
    parser.add_argument('--n-splits', type=int, default=5)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading windows metadata...")
    windows = pd.read_csv(args.windows)
    print(f"  {len(windows)} windows")

    print("Loading window sequences...")
    seqs = pd.read_csv(args.sequences)
    seqs = seqs.set_index('window_id').loc[windows['window_id']].reset_index()
    assert len(seqs) == len(windows), "Sequence/metadata length mismatch"
    print(f"  Aligned {len(seqs)} sequences with metadata")

    print(f"Loading top {args.n_top_motifs} motifs from enrichment file...")
    motifs_df = pd.read_csv(args.motifs)
    top = motifs_df.head(args.n_top_motifs)
    motifs = top['motif'].tolist()
    weights = top['log2_enrichment'].values
    print(f"  Top motif: {motifs[0]} (log2={weights[0]:.3f})")
    print(f"  Last of top {args.n_top_motifs}: {motifs[-1]} (log2={weights[-1]:.3f})")

    print("Building motif presence features...")
    X = build_motif_features(seqs['sequence'], motifs)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Mean presence per motif: {X.mean(axis=0).mean():.3f}")

    print("Computing transparent scores...")
    scores = transparent_score(X, weights)
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    y = (windows['label'] == 'ISV').astype(int).values
    print(f"  Class distribution: ARB={(y == 0).sum()}, ISV={(y == 1).sum()}")

    results = {}

    for scheme_name, group_col in [('accession', 'accession'),
                                   ('species', 'species')]:
        groups = windows[group_col].values
        print(f"\n{'=' * 60}")
        print(f"SCHEME: {scheme_name}-grouped CV")
        print(f"{'=' * 60}")
        print(f"  Unique groups: {len(np.unique(groups))}")

        print(f"\n  (a) Transparent score (sum of weighted motif presence):")
        r_score = evaluate_score(scores, y, groups, n_splits=args.n_splits)
        print(f"      AP  = {r_score['ap_mean']:.4f} ± {r_score['ap_std']:.4f}")
        print(f"      AUC = {r_score['auc_mean']:.4f} ± {r_score['auc_std']:.4f}")
        print(f"      Per-fold AP: {[f'{x:.3f}' for x in r_score['aps']]}")

        print(f"\n  (b) Logistic regression on top-{args.n_top_motifs} motif presence:")
        r_lr = evaluate_logreg(X.astype(float), y, groups, n_splits=args.n_splits)
        print(f"      AP  = {r_lr['ap_mean']:.4f} ± {r_lr['ap_std']:.4f}")
        print(f"      AUC = {r_lr['auc_mean']:.4f} ± {r_lr['auc_std']:.4f}")
        print(f"      Per-fold AP: {[f'{x:.3f}' for x in r_lr['aps']]}")

        results[scheme_name] = {
            'transparent_score': r_score,
            'logreg_top_motifs': r_lr,
        }

    out_file = out / 'motif_scorer_results.json'
    with open(out_file, 'w') as f:
        json.dump({
            'n_top_motifs': args.n_top_motifs,
            'top_motifs': motifs,
            'top_motif_weights': weights.tolist(),
            'results': results,
            'n_windows': int(len(windows)),
            'class_distribution': {'ARB': int((y == 0).sum()),
                                   'ISV': int((y == 1).sum())},
        }, f, indent=2)
    print(f"\nResults written to {out_file}")

    print("\n" + "=" * 60)
    print("COMPARISON TO DNABERT-2 (from all_baseline_results.json)")
    print("=" * 60)
    print(f"  DNABERT-2:")
    print(f"    Accession-grouped: AP = 0.981 ± 0.003")
    print(f"    Species-grouped:   AP = 0.819 ± 0.146")
    print(f"  Motif transparent score:")
    print(f"    Accession-grouped: AP = {results['accession']['transparent_score']['ap_mean']:.3f} ± {results['accession']['transparent_score']['ap_std']:.3f}")
    print(f"    Species-grouped:   AP = {results['species']['transparent_score']['ap_mean']:.3f} ± {results['species']['transparent_score']['ap_std']:.3f}")
    print(f"  Motif logreg top-{args.n_top_motifs}:")
    print(f"    Accession-grouped: AP = {results['accession']['logreg_top_motifs']['ap_mean']:.3f} ± {results['accession']['logreg_top_motifs']['ap_std']:.3f}")
    print(f"    Species-grouped:   AP = {results['species']['logreg_top_motifs']['ap_mean']:.3f} ± {results['species']['logreg_top_motifs']['ap_std']:.3f}")


if __name__ == '__main__':
    main()
