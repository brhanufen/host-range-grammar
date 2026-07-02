"""
eval_utils.py
=============
Evaluation metrics and cross-validation utilities.
"""

from typing import Dict, List, Optional, Tuple, Union

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


def compute_metrics(y_true: np.ndarray, 
                    y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_proba : np.ndarray
        Predicted probabilities for positive class
    
    Returns
    -------
    dict
        Dictionary with AP, ROC-AUC, and optimal threshold
    """
    ap = average_precision_score(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    # Find optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'AP': ap,
        'ROC_AUC': auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity_at_optimal': tpr[optimal_idx],
        'specificity_at_optimal': 1 - fpr[optimal_idx]
    }


def grouped_cv_evaluation(X: np.ndarray,
                          y: np.ndarray,
                          groups: np.ndarray,
                          n_splits: int = 5,
                          random_state: int = 42) -> Dict:
    """
    Perform grouped cross-validation with logistic regression.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Binary labels
    groups : np.ndarray
        Group assignments for samples
    n_splits : int
        Number of CV folds (default: 5)
    random_state : int
        Random seed for reproducibility
    
    Returns
    -------
    dict
        Dictionary containing:
        - oof_proba: out-of-fold predictions
        - fold_metrics: per-fold metrics DataFrame
        - overall_metrics: aggregated metrics
        - curves: PR and ROC curve data
    """
    gkf = GroupKFold(n_splits=n_splits)
    
    oof_proba = np.full(len(y), np.nan)
    fold_metrics = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Check both classes present
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        
        # Train classifier
        clf = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=random_state
        )
        clf.fit(X_train, y_train)
        
        # Predict
        proba = clf.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = proba
        
        # Metrics
        metrics = compute_metrics(y_test, proba)
        metrics['fold'] = fold
        metrics['n_train'] = len(train_idx)
        metrics['n_test'] = len(test_idx)
        metrics['n_pos_train'] = y_train.sum()
        metrics['n_pos_test'] = y_test.sum()
        fold_metrics.append(metrics)
    
    fold_metrics_df = pd.DataFrame(fold_metrics)
    
    # Overall metrics on valid predictions
    valid_mask = ~np.isnan(oof_proba)
    overall = compute_metrics(y[valid_mask], oof_proba[valid_mask])
    
    # Curves for plotting
    precision, recall, _ = precision_recall_curve(y[valid_mask], oof_proba[valid_mask])
    fpr, tpr, _ = roc_curve(y[valid_mask], oof_proba[valid_mask])
    
    return {
        'oof_proba': oof_proba,
        'fold_metrics': fold_metrics_df,
        'overall_metrics': overall,
        'curves': {
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'tpr': tpr
        },
        'summary': {
            'AP_mean': fold_metrics_df['AP'].mean(),
            'AP_std': fold_metrics_df['AP'].std(),
            'AUC_mean': fold_metrics_df['ROC_AUC'].mean(),
            'AUC_std': fold_metrics_df['ROC_AUC'].std()
        }
    }


def compare_evaluation_schemes(X: np.ndarray,
                                y: np.ndarray,
                                grouping_schemes: Dict[str, np.ndarray],
                                n_splits: int = 5) -> pd.DataFrame:
    """
    Compare multiple grouping schemes for cross-validation.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    grouping_schemes : dict
        {scheme_name: group_array}
    n_splits : int
        Number of CV folds
    
    Returns
    -------
    pd.DataFrame
        Comparison of metrics across schemes
    """
    results = []
    
    for scheme_name, groups in grouping_schemes.items():
        cv_result = grouped_cv_evaluation(X, y, groups, n_splits=n_splits)
        
        results.append({
            'scheme': scheme_name,
            'n_groups': len(np.unique(groups)),
            **cv_result['summary']
        })
    
    return pd.DataFrame(results)


def compute_leakage_gap(results_df: pd.DataFrame,
                        baseline_scheme: str = 'accession',
                        stringent_scheme: str = 'species') -> float:
    """
    Compute the evaluation leakage gap between two schemes.
    
    Returns
    -------
    float
        Difference in AP (baseline - stringent)
    """
    baseline_ap = results_df[results_df['scheme'] == baseline_scheme]['AP_mean'].values[0]
    stringent_ap = results_df[results_df['scheme'] == stringent_scheme]['AP_mean'].values[0]
    
    return baseline_ap - stringent_ap
