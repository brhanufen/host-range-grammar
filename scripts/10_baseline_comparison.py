#!/usr/bin/env python3
"""
10_baseline_comparison.py
=========================
Evaluate baseline feature representations (k-mer TF-IDF, composition-only,
position-only) under accession-grouped and species-grouped cross-validation.

Demonstrates that evaluation leakage is method-agnostic.

Inputs:
    data/windows_metadata.csv
    data/windows_sequences.csv
    embeddings/embeddings_combined.npy

Outputs:
    results/all_baseline_results.json
    results/oof_predictions.npz
"""

import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_dinucleotide_features(sequences):
    """Compute GC content + 16 dinucleotide frequencies."""
    dinucs = [a + b for a in 'ACGT' for b in 'ACGT']
    features = []
    for seq in sequences:
        seq = seq.upper()
        n = len(seq)
        gc = (seq.count('G') + seq.count('C')) / n if n > 0 else 0
        dinuc_freq = []
        for di in dinucs:
            count = sum(1 for i in range(n - 1) if seq[i:i+2] == di)
            dinuc_freq.append(count / (n - 1) if n > 1 else 0)
        features.append([gc] + dinuc_freq)
    return np.array(features)


def kmer_to_spaced(seq, k=6):
    """Convert sequence to space-separated k-mers for TfidfVectorizer."""
    return ' '.join(seq[i:i+k] for i in range(len(seq) - k + 1))


def run_grouped_cv(X, y, groups, n_splits=5, random_state=42):
    """Run grouped cross-validation with logistic regression.
    Returns per-fold AP (ISV as positive class) and OOF predictions."""
    gkf = GroupKFold(n_splits=n_splits)
    aps, aucs = [], []
    oof_proba = np.full(len(y), np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(class_weight='balanced', max_iter=1000,
                                 random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        oof_proba[test_idx] = proba

        # ISV as positive class: invert labels and probabilities
        ap = average_precision_score(1 - y[test_idx], 1 - proba)
        auc = roc_auc_score(1 - y[test_idx], 1 - proba)
        aps.append(ap)
        aucs.append(auc)

    return {
        'ap_mean': float(np.mean(aps)),
        'ap_std': float(np.std(aps)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
        'aps': [float(a) for a in aps],
    }, oof_proba


def main():
    print("=" * 70)
    print("BASELINE COMPARISON: EVALUATION LEAKAGE IS METHOD-AGNOSTIC")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading data...")
    windows = pd.read_csv('data/windows_metadata.csv')
    sequences = pd.read_csv('data/windows_sequences.csv')
    embeddings = np.load('embeddings/embeddings_combined.npy')

    y_arb = (windows['label'] == 'ARB').astype(int).values
    y_isv = 1 - y_arb  # ISV = positive class
    groups_acc = windows['accession'].values
    groups_sp = windows['species'].values

    print(f"  Windows: {len(windows)}")
    print(f"  ARB: {y_arb.sum()}, ISV: {y_isv.sum()}")

    results = {}
    oof_dict = {'y_isv': y_isv}

    # ── DNABERT-2 ──
    print("\n[2/6] DNABERT-2 embeddings...")
    X_emb = embeddings
    np.random.seed(42)

    for scheme, groups, suffix in [
        ('random', np.arange(len(y_arb)), 'rand'),
        ('accession', groups_acc, 'acc'),
        ('species', groups_sp, 'sp'),
    ]:
        res, oof = run_grouped_cv(X_emb, y_arb, groups)
        results.setdefault('DNABERT-2', {})[scheme] = res
        oof_dict[f'dnabert2_{suffix}'] = oof
        print(f"  {scheme}: AP = {res['ap_mean']:.3f} +/- {res['ap_std']:.3f}")

    # ── k-mer TF-IDF ──
    print("\n[3/6] k-mer TF-IDF (6-mers)...")
    seq_list = sequences['sequence'].tolist()
    spaced = [kmer_to_spaced(s) for s in seq_list]
    vectorizer = TfidfVectorizer(analyzer='word', lowercase=False)
    X_tfidf = vectorizer.fit_transform(spaced)

    for scheme, groups, suffix in [
        ('accession', groups_acc, 'acc'),
        ('species', groups_sp, 'sp'),
    ]:
        res, oof = run_grouped_cv(X_tfidf, y_arb, groups)
        results.setdefault('k-mer TF-IDF', {})[scheme] = res
        oof_dict[f'tfidf_{suffix}'] = oof
        print(f"  {scheme}: AP = {res['ap_mean']:.3f} +/- {res['ap_std']:.3f}")

    # ── Composition-only ──
    print("\n[4/6] Composition features (GC + 16 dinucleotides)...")
    X_comp = compute_dinucleotide_features(seq_list)
    print(f"  Feature dimensions: {X_comp.shape[1]}")

    for scheme, groups, suffix in [
        ('accession', groups_acc, 'acc'),
        ('species', groups_sp, 'sp'),
    ]:
        res, oof = run_grouped_cv(X_comp, y_arb, groups)
        results.setdefault('Composition', {})[scheme] = res
        oof_dict[f'comp_{suffix}'] = oof
        print(f"  {scheme}: AP = {res['ap_mean']:.3f} +/- {res['ap_std']:.3f}")

    # ── Position-only ──
    print("\n[5/6] Position-only (genome coordinate)...")
    X_pos = windows[['start']].values.astype(float)

    for scheme, groups, suffix in [
        ('accession', groups_acc, 'acc'),
        ('species', groups_sp, 'sp'),
    ]:
        res, _ = run_grouped_cv(X_pos, y_arb, groups)
        results.setdefault('Position-only', {})[scheme] = res
        print(f"  {scheme}: AP = {res['ap_mean']:.3f} +/- {res['ap_std']:.3f}")

    # ── Save ──
    print("\n[6/6] Saving results...")
    with open('results/all_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez('results/oof_predictions.npz', **oof_dict)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Method':<20} {'Acc-grouped AP':>16} {'Sp-grouped AP':>16} {'Drop (pp)':>10}")
    print("-" * 70)
    for method in ['DNABERT-2', 'k-mer TF-IDF', 'Composition', 'Position-only']:
        r = results[method]
        ap_a = r['accession']['ap_mean']
        ap_s = r['species']['ap_mean']
        drop = (ap_a - ap_s) * 100
        print(f"{method:<20} {ap_a:>11.3f} +/- {r['accession']['ap_std']:.3f}"
              f" {ap_s:>6.3f} +/- {r['species']['ap_std']:.3f} {drop:>8.1f}")
    print("=" * 70)
    print("Saved: results/all_baseline_results.json")
    print("Saved: results/oof_predictions.npz")


if __name__ == '__main__':
    main()
