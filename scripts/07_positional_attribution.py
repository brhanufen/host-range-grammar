#!/usr/bin/env python3
"""
07_positional_attribution.py
============================
Genome-localized attribution analysis using window-level predictions.

Identifies positional hotspots where discriminative signal concentrates.

Usage:
    python scripts/07_positional_attribution.py \
        --predictions results/window_predictions.csv \
        --output results/

Author: [Author Name]
Date: 2026
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_positional_enrichment(windows_df: pd.DataFrame,
                                   score_col: str = 'p_arb_accession',
                                   n_bins: int = 20,
                                   top_percentile: float = 0.10) -> pd.DataFrame:
    """
    Compute positional enrichment of high-scoring windows.
    
    Parameters
    ----------
    windows_df : pd.DataFrame
        Window predictions with position and score columns
    score_col : str
        Column name for prediction scores
    n_bins : int
        Number of positional bins (default: 20)
    top_percentile : float
        Fraction of top-scoring windows (default: 0.10 = top 10%)
    
    Returns
    -------
    pd.DataFrame
        Enrichment by position bin and label
    """
    # Compute threshold for "top" windows
    threshold = windows_df[score_col].quantile(1 - top_percentile)
    windows_df['is_top'] = windows_df[score_col] >= threshold
    
    # Assign position bins (0-indexed)
    windows_df['pos_bin'] = (windows_df['pos_rel'] * n_bins).astype(int).clip(0, n_bins - 1)
    
    results = []
    
    for label in ['ARB', 'ISV']:
        subset = windows_df[windows_df['label'] == label]
        top_subset = subset[subset['is_top']]
        
        # Expected fraction per bin (uniform)
        expected = 1.0 / n_bins
        
        for bin_idx in range(n_bins):
            # Count windows in this bin
            n_total = (subset['pos_bin'] == bin_idx).sum()
            n_top = (top_subset['pos_bin'] == bin_idx).sum()
            
            # Observed fraction of top windows in this bin
            observed = n_top / len(top_subset) if len(top_subset) > 0 else 0
            
            # Enrichment ratio
            enrichment = observed / expected if expected > 0 else 0
            
            # Bin midpoint
            midpoint = (bin_idx + 0.5) / n_bins
            
            results.append({
                'label': label,
                'bin': bin_idx,
                'midpoint': midpoint,
                'n_total': n_total,
                'n_top': n_top,
                'observed_frac': observed,
                'expected_frac': expected,
                'enrichment': enrichment
            })
    
    return pd.DataFrame(results)


def compute_genome_hotspot_coverage(windows_df: pd.DataFrame,
                                     score_col: str = 'p_arb_accession',
                                     n_bins: int = 20,
                                     top_percentile: float = 0.10) -> pd.DataFrame:
    """
    Compute fraction of genomes with at least one top-scoring window per bin.
    
    Returns
    -------
    pd.DataFrame
        Genome coverage by position bin and label
    """
    threshold = windows_df[score_col].quantile(1 - top_percentile)
    windows_df['is_top'] = windows_df[score_col] >= threshold
    windows_df['pos_bin'] = (windows_df['pos_rel'] * n_bins).astype(int).clip(0, n_bins - 1)
    
    results = []
    
    for label in ['ARB', 'ISV']:
        subset = windows_df[windows_df['label'] == label]
        total_genomes = subset['accession'].nunique()
        
        for bin_idx in range(n_bins):
            # Genomes with at least one top window in this bin
            bin_windows = subset[(subset['pos_bin'] == bin_idx) & subset['is_top']]
            genomes_with_top = bin_windows['accession'].nunique()
            
            coverage = genomes_with_top / total_genomes if total_genomes > 0 else 0
            
            results.append({
                'label': label,
                'bin': bin_idx,
                'midpoint': (bin_idx + 0.5) / n_bins,
                'genomes_with_top': genomes_with_top,
                'total_genomes': total_genomes,
                'coverage': coverage
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Positional attribution analysis'
    )
    parser.add_argument(
        '--predictions', required=True,
        help='Window predictions CSV file'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--n-bins', type=int, default=20,
        help='Number of positional bins (default: 20)'
    )
    parser.add_argument(
        '--top-percentile', type=float, default=0.10,
        help='Top percentile for enrichment (default: 0.10)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    windows_df = pd.read_csv(args.predictions)
    print(f"  {len(windows_df)} windows")
    
    # Compute positional enrichment
    print(f"\nComputing positional enrichment (n_bins={args.n_bins})...")
    enrichment_df = compute_positional_enrichment(
        windows_df.copy(),
        n_bins=args.n_bins,
        top_percentile=args.top_percentile
    )
    enrichment_df.to_csv(output_dir / "positional_enrichment.csv", index=False)
    
    # Compute genome coverage
    print("Computing genome hotspot coverage...")
    coverage_df = compute_genome_hotspot_coverage(
        windows_df.copy(),
        n_bins=args.n_bins,
        top_percentile=args.top_percentile
    )
    coverage_df.to_csv(output_dir / "genome_hotspot_coverage.csv", index=False)
    
    # Find peak enrichment positions
    print("\n" + "="*60)
    print("POSITIONAL HOTSPOT SUMMARY")
    print("="*60)
    
    for label in ['ARB', 'ISV']:
        label_enrich = enrichment_df[enrichment_df['label'] == label]
        peak = label_enrich.loc[label_enrich['enrichment'].idxmax()]
        
        print(f"\n{label}:")
        print(f"  Peak enrichment:  {peak['enrichment']:.2f}x at position {peak['midpoint']:.3f}")
        print(f"  Peak bin:         {int(peak['bin'])} (midpoint = {peak['midpoint']:.3f})")
        
        # Top 3 bins
        top3 = label_enrich.nlargest(3, 'enrichment')
        print(f"  Top 3 bins:       {', '.join([str(int(b)) for b in top3['bin']])}")
    
    # Genome coverage summary
    print("\nGenome coverage at peak hotspots:")
    for label in ['ARB', 'ISV']:
        label_cov = coverage_df[coverage_df['label'] == label]
        
        # Coverage in bins 13-15 (NS3-NS5 region)
        hotspot_cov = label_cov[label_cov['bin'].isin([13, 14, 15])]['coverage'].mean()
        print(f"  {label} at bins 13-15: {hotspot_cov*100:.1f}% of genomes")
    
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
