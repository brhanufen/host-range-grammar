#!/usr/bin/env python3
"""
05_grouped_cv.py
================
Lineage-aware grouped cross-validation for ARB vs ISV classification.

Implements two evaluation schemes:
1. Accession-grouped: Windows from same genome in same fold
2. Species-grouped: All genomes of same species in same fold

Usage:
    python scripts/05_grouped_cv.py \
        --embeddings data/embeddings/window_embeddings.npy \
        --windows data/windows/windows_metadata.csv \
        --output results/

Author: [Author Name]
Date: 2026
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

warnings.filterwarnings('ignore')


def evaluate_grouped_cv(X: np.ndarray, 
                        y: np.ndarray, 
                        groups: np.ndarray,
                        n_splits: int = 5) -> dict:
    """
    Perform grouped cross-validation and return metrics.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels (0=ARB, 1=ISV)
    groups : np.ndarray
        Group assignments for each sample
    n_splits : int
        Number of CV folds (default: 5)
    
    Returns
    -------
    dict
        Dictionary with OOF predictions, metrics per fold, and curves
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_proba = np.zeros(len(y))
    oof_mask = np.zeros(len(y), dtype=bool)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check both classes present
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"  Fold {fold}: Skipping (missing class)")
            continue
        
        # Train classifier
        clf = LogisticRegression(
            max_iter=1000, 
            class_weight='balanced',
            random_state=42
        )
        clf.fit(X_train, y_train)
        
        # Predict probabilities
        proba = clf.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba
        oof_mask[test_idx] = True
        
        # Compute metrics
        ap = average_precision_score(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        
        fold_metrics.append({
            'fold': fold,
            'AP': ap,
            'ROC_AUC': auc,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_pos_test': y_test.sum()
        })
        
        print(f"  Fold {fold}: AP={ap:.4f}, AUC={auc:.4f}")
    
    # Compute overall metrics on valid OOF predictions
    valid_idx = oof_mask
    overall_ap = average_precision_score(y[valid_idx], oof_proba[valid_idx])
    overall_auc = roc_auc_score(y[valid_idx], oof_proba[valid_idx])
    
    # Compute curves for plotting
    precision, recall, pr_thresholds = precision_recall_curve(
        y[valid_idx], oof_proba[valid_idx]
    )
    fpr, tpr, roc_thresholds = roc_curve(
        y[valid_idx], oof_proba[valid_idx]
    )
    
    return {
        'oof_proba': oof_proba,
        'oof_mask': oof_mask,
        'fold_metrics': pd.DataFrame(fold_metrics),
        'overall_AP': overall_ap,
        'overall_AUC': overall_auc,
        'precision': precision,
        'recall': recall,
        'fpr': fpr,
        'tpr': tpr
    }


def main():
    parser = argparse.ArgumentParser(
        description='Lineage-aware grouped cross-validation'
    )
    parser.add_argument(
        '--embeddings', required=True,
        help='Window embeddings NPY file'
    )
    parser.add_argument(
        '--windows', required=True,
        help='Window metadata CSV file'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-splits', type=int, default=5,
        help='Number of CV folds (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    embeddings = np.load(args.embeddings)
    windows_df = pd.read_csv(args.windows)
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Windows: {len(windows_df)}")
    
    # Prepare labels (ISV = 1, ARB = 0)
    y = (windows_df['label'] == 'ISV').astype(int).values
    print(f"  Class distribution: ARB={sum(y==0)}, ISV={sum(y==1)}")
    
    # Define grouping schemes
    grouping_schemes = {
        'accession': windows_df['accession'].values,
        'species': windows_df['species'].values
    }
    
    all_results = {}
    
    for scheme_name, groups in grouping_schemes.items():
        print(f"\n{'='*60}")
        print(f"EVALUATION: {scheme_name}-grouped cross-validation")
        print(f"{'='*60}")
        print(f"Number of unique groups: {len(np.unique(groups))}")
        
        results = evaluate_grouped_cv(
            embeddings, y, groups, n_splits=args.n_splits
        )
        all_results[scheme_name] = results
        
        # Save fold metrics
        fold_file = output_dir / f"cv_results_{scheme_name}.csv"
        results['fold_metrics'].to_csv(fold_file, index=False)
        
        # Summary statistics
        metrics_df = results['fold_metrics']
        print(f"\nSummary:")
        print(f"  AP:  {metrics_df['AP'].mean():.4f} ± {metrics_df['AP'].std():.4f}")
        print(f"  AUC: {metrics_df['ROC_AUC'].mean():.4f} ± {metrics_df['ROC_AUC'].std():.4f}")
    
    # Save OOF predictions for attribution
    print("\n" + "="*60)
    print("Saving out-of-fold predictions...")
    
    windows_df['p_isv_accession'] = all_results['accession']['oof_proba']
    windows_df['p_isv_species'] = all_results['species']['oof_proba']
    windows_df['p_arb_accession'] = 1 - windows_df['p_isv_accession']
    windows_df['p_arb_species'] = 1 - windows_df['p_isv_species']
    
    oof_file = output_dir / "window_predictions.csv"
    windows_df.to_csv(oof_file, index=False)
    
    # Save PR curves for plotting
    for scheme_name, results in all_results.items():
        curve_file = output_dir / f"pr_curve_{scheme_name}.csv"
        curve_df = pd.DataFrame({
            'precision': results['precision'],
            'recall': results['recall']
        })
        curve_df.to_csv(curve_file, index=False)
    
    # Create summary comparison
    summary_data = []
    for scheme_name, results in all_results.items():
        metrics_df = results['fold_metrics']
        summary_data.append({
            'scheme': scheme_name,
            'AP_mean': metrics_df['AP'].mean(),
            'AP_std': metrics_df['AP'].std(),
            'AUC_mean': metrics_df['ROC_AUC'].mean(),
            'AUC_std': metrics_df['ROC_AUC'].std()
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / "evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Compute leakage gap
    acc_ap = all_results['accession']['fold_metrics']['AP'].mean()
    spe_ap = all_results['species']['fold_metrics']['AP'].mean()
    gap = acc_ap - spe_ap
    
    print(f"\nEvaluation leakage gap: {gap*100:.1f} percentage points")
    print(f"  (Accession AP: {acc_ap:.3f} → Species AP: {spe_ap:.3f})")
    
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
