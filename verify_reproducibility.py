#!/usr/bin/env python3
"""
Verify all key manuscript values are reproducible.
Run this script to confirm your environment and data are set up correctly.

Usage:
    python verify_reproducibility.py
"""

import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


def main():
    print("=" * 70)
    print("REPRODUCIBILITY VERIFICATION")
    print("=" * 70)

    checks = []

    # ── [1] Load data ──
    print("\n[1/8] Loading data...")
    meta = pd.read_csv('data/metadata.csv')
    windows = pd.read_csv('data/windows_metadata.csv')
    embeddings = np.load('embeddings/embeddings_combined.npy')

    print(f"  Genomes: {len(meta)} (expected: 3,031)")
    print(f"  Windows: {len(windows)} (expected: 59,980)")
    print(f"  Embeddings: {embeddings.shape} (expected: (59980, 768))")

    checks.append(("Genomes = 3,031", len(meta) == 3031))
    checks.append(("Windows = 59,980", len(windows) == 59980))
    checks.append(("Embeddings = (59980, 768)", embeddings.shape == (59980, 768)))

    n_arb = (meta['label'] == 'ARB').sum()
    n_isv = (meta['label'] == 'ISV').sum()
    print(f"  ARB genomes: {n_arb} (expected: 2,816)")
    print(f"  ISV genomes: {n_isv} (expected: 215)")
    checks.append(("ARB = 2,816", n_arb == 2816))
    checks.append(("ISV = 215", n_isv == 215))

    # Prepare data
    X = embeddings
    y = (windows['label'] == 'ARB').astype(int).values
    groups_acc = windows['accession'].values
    groups_sp = windows['species'].values
    np.random.seed(42)
    gkf = GroupKFold(n_splits=5)

    # ── [2] DNABERT-2 accession-grouped CV ──
    print("\n[2/8] DNABERT-2 accession-grouped CV...")
    aps_acc = []
    for train_idx, test_idx in gkf.split(X, y, groups_acc):
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aps_acc.append(average_precision_score(1 - y[test_idx], 1 - proba))
    ap_acc = np.mean(aps_acc)
    print(f"  AP = {ap_acc:.3f} (expected: 0.981)")
    checks.append(("DNABERT-2 acc AP ~ 0.981", abs(ap_acc - 0.981) < 0.01))

    # ── [3] DNABERT-2 species-grouped CV ──
    print("\n[3/8] DNABERT-2 species-grouped CV...")
    aps_sp = []
    for train_idx, test_idx in gkf.split(X, y, groups_sp):
        clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aps_sp.append(average_precision_score(1 - y[test_idx], 1 - proba))
    ap_sp = np.mean(aps_sp)
    drop = (ap_acc - ap_sp) * 100
    print(f"  AP = {ap_sp:.3f} (expected: 0.819)")
    print(f"  Drop = {drop:.1f} pp (expected: ~16 pp)")
    checks.append(("DNABERT-2 sp AP ~ 0.819", abs(ap_sp - 0.819) < 0.05))
    checks.append(("Leakage drop ~ 16 pp", abs(drop - 16) < 5))

    # ── [4] Embedding space metrics ──
    print("\n[4/8] Embedding space metrics...")
    genome_embs, genome_labels = [], []
    for acc in meta['accession'].values:
        mask = windows['accession'] == acc
        if mask.sum() > 0:
            idx = windows[mask].index.values
            genome_embs.append(embeddings[idx].mean(axis=0))
            genome_labels.append(meta[meta['accession'] == acc]['label'].values[0])
    genome_embs = np.array(genome_embs)
    genome_labels = np.array(genome_labels)

    nn = NearestNeighbors(n_neighbors=16)
    nn.fit(genome_embs)
    _, indices = nn.kneighbors(genome_embs)
    purities = [(genome_labels[indices[i, 1:]] == genome_labels[i]).mean()
                for i in range(len(genome_labels))]
    knn_purity = np.mean(purities)
    sil = silhouette_score(genome_embs, (genome_labels == 'ARB').astype(int))

    print(f"  kNN Purity (k=15) = {knn_purity:.3f} (expected: 0.997)")
    print(f"  Silhouette Score  = {sil:.3f} (expected: 0.433)")
    checks.append(("kNN purity ~ 0.997", abs(knn_purity - 0.997) < 0.005))
    checks.append(("Silhouette ~ 0.433", abs(sil - 0.433) < 0.02))

    # ── [5] Baseline results ──
    print("\n[5/8] Checking baseline results...")
    try:
        with open('results/all_baseline_results.json') as f:
            baselines = json.load(f)
        for method, exp_acc, exp_sp in [
            ('k-mer TF-IDF', 0.997, 0.858),
            ('Composition', 0.928, 0.834),
            ('Position-only', 0.071, 0.070),
        ]:
            r = baselines[method]
            ap_a = r['accession']['ap_mean']
            ap_s = r['species']['ap_mean']
            print(f"  {method}: acc={ap_a:.3f} (exp {exp_acc:.3f}), "
                  f"sp={ap_s:.3f} (exp {exp_sp:.3f})")
            checks.append((f"{method} acc AP", abs(ap_a - exp_acc) < 0.01))
            checks.append((f"{method} sp AP", abs(ap_s - exp_sp) < 0.05))
    except FileNotFoundError:
        print("  results/all_baseline_results.json not found.")
        print("  Run: python scripts/10_baseline_comparison.py")
        checks.append(("Baseline results file exists", False))

    # ── [6] Motif enrichment ──
    print("\n[6/8] Checking motif enrichment results...")
    try:
        motifs = pd.read_csv('results/motif_enrichment_full.csv')
        top10 = motifs.nlargest(10, 'log2_enrichment')
        n_upa_no_cpg = ((top10['n_UpA'] > 0) & (top10['n_CpG'] == 0)).sum()
        print(f"  Total 6-mers: {len(motifs)} (expected: 4,095; 1 of 4,096 excluded by min-prevalence filter)")
        print(f"  Top motif: {top10.iloc[0]['motif']} (expected: ATTAGG)")
        print(f"  UpA-only in top 10: {n_upa_no_cpg} (expected: 6)")
        checks.append(("6-mer count = 4,095", len(motifs) == 4095))
        checks.append(("Top motif = ATTAGG", top10.iloc[0]['motif'] == 'ATTAGG'))
        checks.append(("UpA-only in top 10 = 6", n_upa_no_cpg == 6))
    except FileNotFoundError:
        print("  results/motif_enrichment_full.csv not found.")
        print("  Run: python scripts/08_motif_enrichment.py")
        checks.append(("Motif enrichment file exists", False))


    # ── [7] Composition-feature separation (Supplementary Figure S1 / Table S2) ──
    print("\n[7/8] Checking composition-feature separation (Suppl. Fig S1)...")
    try:
        from scipy.stats import mannwhitneyu
        seqs = pd.read_csv('data/windows_sequences.csv')
        wmS1 = pd.read_csv('data/windows_metadata.csv')[['window_id', 'label']]
        dS1 = seqs.merge(wmS1, on='window_id', how='inner')
        lab = dS1['label'].values
        arb_m = (lab == 'ARB')
        isv_m = ~arb_m
        nA, nI = int(arb_m.sum()), int(isv_m.sum())

        def _dinuc_freq(sequences, di):
            out = np.empty(len(sequences))
            for j, s in enumerate(sequences):
                s = s.upper(); n = len(s)
                out[j] = (sum(1 for i in range(n - 1) if s[i:i+2] == di) / (n - 1)) if n > 1 else 0.0
            return out

        seq_vals = dS1['sequence'].values
        cg = _dinuc_freq(seq_vals, 'CG')
        ta = _dinuc_freq(seq_vals, 'TA')

        def _auc_d(x):
            a, i = x[arb_m], x[isv_m]
            U, _ = mannwhitneyu(i, a, alternative='two-sided')
            auc = U / (nA * nI)
            sp = np.sqrt(((nA - 1) * a.var(ddof=1) + (nI - 1) * i.var(ddof=1)) / (nA + nI - 2))
            d = (i.mean() - a.mean()) / sp if sp > 0 else 0.0
            return auc, d

        cg_auc, cg_d = _auc_d(cg)
        ta_auc, ta_d = _auc_d(ta)
        print(f"  CpG separation: AUC = {cg_auc:.3f} (expected: 0.943), Cohen's d = {cg_d:.2f} (expected: 2.71)")
        print(f"  UpA separation: AUC = {ta_auc:.3f} (expected: 0.524), Cohen's d = {ta_d:.2f} (expected: 0.07)")
        checks.append(("CpG is top discriminator (AUC ~ 0.94)", abs(cg_auc - 0.943) < 0.01))
        checks.append(("UpA minimal composition separation (AUC ~ 0.52)", abs(ta_auc - 0.524) < 0.01))
    except FileNotFoundError:
        print("  data/windows_sequences.csv not found.")
        print("  Run: python scripts/figures/regen_figS1.py")
        checks.append(("Composition-feature data exists", False))

    # ── [8] Summary ──
    print("\n[8/8] VERIFICATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in checks:
        symbol = "+" if passed else "X"
        status = "PASS" if passed else "FAIL"
        print(f"  [{symbol}] {status}: {name}")
        if not passed:
            all_pass = False

    n_pass = sum(1 for _, p in checks if p)
    n_total = len(checks)
    print(f"\n  {n_pass}/{n_total} checks passed")
    print("=" * 70)
    if all_pass:
        print("ALL CHECKS PASSED - FULLY REPRODUCIBLE")
    else:
        print(f"{n_total - n_pass} CHECK(S) FAILED - Please verify your setup")
    print("=" * 70)


if __name__ == "__main__":
    main()
