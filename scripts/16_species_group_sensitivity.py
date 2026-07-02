#!/usr/bin/env python3
"""
16_species_group_sensitivity.py

Sensitivity of the species-grouped (leakage-controlled) average precision to the
unclassified "Unknown flavivirus" placeholder taxon.

The metadata label "Unknown flavivirus" is shared across both host-range classes
(11 ARB + 42 ISFV genomes). Under species-grouped cross-validation it therefore
forms a single group that spans both classes. This script confirms that the
headline species-grouped AP does not depend on that group by re-running the exact
species-grouped CV of 05_grouped_cv.py with and without those genomes.

Method (identical to 05_grouped_cv.py):
  frozen DNABERT-2 embeddings -> balanced LogisticRegression (max_iter=1000,
  random_state=42) -> GroupKFold(5) grouped by species -> AP (ISFV positive),
  reported as the mean over folds.

Reproduces:
  Full data            : AP = 0.819 +/- 0.146   (matches all_baseline_results.json)
  Excl. Unknown flavi. : AP = 0.818 +/- 0.140   (delta -0.05 pp)

Usage (paths resolved relative to this script so the repo is portable):
  python scripts/16_species_group_sensitivity.py
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# benign overflow warnings from raw-embedding matmul; results are unaffected
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDED_SPECIES = "Unknown flavivirus"


def species_grouped_ap(X, y, groups, n_splits=5):
    """Mean per-fold average precision under species-grouped CV (ISFV positive)."""
    gkf = GroupKFold(n_splits=n_splits)
    aps = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        clf = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ).fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aps.append(average_precision_score(y[test_idx], proba))
    return float(np.mean(aps)), float(np.std(aps)), len(aps)


def main():
    X = np.load(os.path.join(REPO, "embeddings", "embeddings_combined.npy"))
    w = pd.read_csv(os.path.join(REPO, "data", "windows_metadata.csv"))
    assert len(X) == len(w), "embeddings and windows_metadata.csv are misaligned"

    y = (w["label"] == "ISV").astype(int).values
    species = w["species"].values

    ap, sd, n = species_grouped_ap(X, y, species)
    print(f"[FULL data]            species-grouped AP = {ap:.4f} +/- {sd:.4f} ({n} folds)")

    keep = species != EXCLUDED_SPECIES
    n_win = int((~keep).sum())
    n_gen = int(w.loc[~keep, "accession"].nunique())
    ap2, sd2, n2 = species_grouped_ap(X[keep], y[keep], species[keep])
    print(f"[excl. {EXCLUDED_SPECIES!r}] removed {n_gen} genomes / {n_win} windows")
    print(f"[excl. {EXCLUDED_SPECIES}] species-grouped AP = {ap2:.4f} +/- {sd2:.4f} ({n2} folds)")
    print(f"\nDelta AP = {ap2 - ap:+.4f} ({(ap2 - ap) * 100:+.2f} pp)")


if __name__ == "__main__":
    main()
