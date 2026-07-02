#!/usr/bin/env python3
"""
08_motif_enrichment.py
======================
6-mer motif enrichment analysis comparing top vs bottom scoring windows.

Identifies sequence motifs associated with high ARB prediction scores.

Usage:
    python scripts/08_motif_enrichment.py \
        --predictions results/window_predictions.csv \
        --fasta data/windows/windows.fasta \
        --output results/

Author: [Author Name]
Date: 2026
"""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def count_motif_presence(sequences: dict, motif: str) -> dict:
    """
    Check presence of a motif in each sequence.
    
    Returns
    -------
    dict
        {window_id: True/False}
    """
    presence = {}
    for window_id, seq in sequences.items():
        presence[window_id] = motif in seq
    return presence


def compute_motif_enrichment(windows_df: pd.DataFrame,
                              sequences: dict,
                              score_col: str = 'p_arb_accession',
                              top_frac: float = 0.05,
                              bottom_frac: float = 0.05,
                              k: int = 6,
                              min_prevalence: float = 0.001) -> pd.DataFrame:
    """
    Compute log2 enrichment of k-mers in top vs bottom scoring windows.
    
    Parameters
    ----------
    windows_df : pd.DataFrame
        Window predictions
    sequences : dict
        {window_id: sequence}
    score_col : str
        Prediction score column
    top_frac : float
        Fraction for top stratum (default: 0.05)
    bottom_frac : float
        Fraction for bottom stratum (default: 0.05)
    k : int
        k-mer length (default: 6)
    min_prevalence : float
        Minimum prevalence in bottom stratum to include (default: 0.001)
    
    Returns
    -------
    pd.DataFrame
        Motif enrichment results
    """
    # Define strata
    top_thresh = windows_df[score_col].quantile(1 - top_frac)
    bot_thresh = windows_df[score_col].quantile(bottom_frac)
    
    top_windows = set(windows_df[windows_df[score_col] >= top_thresh]['window_id'])
    bot_windows = set(windows_df[windows_df[score_col] <= bot_thresh]['window_id'])
    
    print(f"  Top stratum: {len(top_windows)} windows (≥{top_thresh:.4f})")
    print(f"  Bottom stratum: {len(bot_windows)} windows (≤{bot_thresh:.4f})")
    
    # Generate all k-mers
    bases = 'ACGT'
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    
    results = []
    
    for motif in tqdm(all_kmers, desc=f"Analyzing {k}-mers"):
        # Count presence in each stratum
        top_count = sum(1 for wid in top_windows if motif in sequences.get(wid, ''))
        bot_count = sum(1 for wid in bot_windows if motif in sequences.get(wid, ''))
        
        top_prev = top_count / len(top_windows) if len(top_windows) > 0 else 0
        bot_prev = bot_count / len(bot_windows) if len(bot_windows) > 0 else 0
        
        # Skip if too rare in bottom stratum
        if bot_prev < min_prevalence:
            continue
        
        # Log2 enrichment
        enrichment = np.log2(top_prev / bot_prev) if bot_prev > 0 else np.nan
        
        # Count dinucleotide content
        n_cg = motif.count('CG')
        n_ua = motif.count('TA')  # TA in DNA = UA in RNA
        
        results.append({
            'motif': motif,
            'top_prevalence': top_prev,
            'bottom_prevalence': bot_prev,
            'log2_enrichment': enrichment,
            'top_count': top_count,
            'bottom_count': bot_count,
            'n_CpG': n_cg,
            'n_UpA': n_ua,
            'gc_content': (motif.count('G') + motif.count('C')) / len(motif)
        })
    
    return pd.DataFrame(results).sort_values('log2_enrichment', ascending=False)


def compute_motif_conservation(sequences_by_genome: dict,
                                genome_df: pd.DataFrame,
                                motifs: list) -> pd.DataFrame:
    """
    Compute conservation (prevalence) of motifs across genomes by class.
    
    Parameters
    ----------
    sequences_by_genome : dict
        {accession: full_genome_sequence}
    genome_df : pd.DataFrame
        Genome metadata with 'accession' and 'label' columns
    motifs : list
        List of motifs to analyze
    
    Returns
    -------
    pd.DataFrame
        Conservation by motif and label
    """
    results = []
    
    for motif in motifs:
        for label in ['ARB', 'ISV']:
            label_genomes = genome_df[genome_df['label'] == label]['accession']
            
            n_present = 0
            n_total = 0
            
            for acc in label_genomes:
                if acc in sequences_by_genome:
                    n_total += 1
                    if motif in sequences_by_genome[acc]:
                        n_present += 1
            
            conservation = n_present / n_total if n_total > 0 else 0
            
            results.append({
                'motif': motif,
                'label': label,
                'n_present': n_present,
                'n_total': n_total,
                'conservation': conservation
            })
    
    return pd.DataFrame(results)


def compute_motif_localization(windows_df: pd.DataFrame,
                                sequences: dict,
                                motifs: list,
                                n_bins: int = 20) -> pd.DataFrame:
    """
    Compute positional localization profile for each motif.
    
    Returns
    -------
    pd.DataFrame
        Motif frequency by position bin
    """
    results = []
    
    for motif in motifs:
        for bin_idx in range(n_bins):
            bin_windows = windows_df[windows_df['pos_bin'] == bin_idx]['window_id']
            
            n_with_motif = sum(1 for wid in bin_windows if motif in sequences.get(wid, ''))
            n_total = len(bin_windows)
            
            freq = n_with_motif / n_total if n_total > 0 else 0
            
            results.append({
                'motif': motif,
                'bin': bin_idx,
                'midpoint': (bin_idx + 0.5) / n_bins,
                'frequency': freq,
                'count': n_with_motif,
                'total': n_total
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='6-mer motif enrichment analysis'
    )
    parser.add_argument(
        '--predictions', required=True,
        help='Window predictions CSV file'
    )
    parser.add_argument(
        '--fasta', required=True,
        help='Window sequences FASTA file'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--top-frac', type=float, default=0.05,
        help='Top fraction for enrichment (default: 0.05)'
    )
    parser.add_argument(
        '--k', type=int, default=6,
        help='k-mer length (default: 6)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading predictions...")
    windows_df = pd.read_csv(args.predictions)
    
    print("Loading sequences...")
    sequences = {}
    for record in SeqIO.parse(args.fasta, 'fasta'):
        sequences[record.id] = str(record.seq).upper()
    print(f"  {len(sequences)} sequences loaded")
    
    # Compute enrichment
    print(f"\nComputing {args.k}-mer enrichment...")
    enrichment_df = compute_motif_enrichment(
        windows_df, 
        sequences,
        top_frac=args.top_frac,
        bottom_frac=args.top_frac,
        k=args.k
    )
    enrichment_df.to_csv(output_dir / "motif_enrichment.csv", index=False)
    
    # Get top motifs for further analysis
    top_motifs = enrichment_df.head(20)['motif'].tolist()
    
    # Motif localization
    print("\nComputing motif localization profiles...")
    
    # Ensure pos_bin is computed
    n_bins = 20
    windows_df['pos_bin'] = (windows_df['pos_rel'] * n_bins).astype(int).clip(0, n_bins - 1)
    
    localization_df = compute_motif_localization(
        windows_df, sequences, top_motifs, n_bins=n_bins
    )
    localization_df.to_csv(output_dir / "motif_localization.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("TOP 10 ENRICHED MOTIFS")
    print("="*60)
    
    top10 = enrichment_df.head(10)
    for _, row in top10.iterrows():
        upa = f"UpA×{row['n_UpA']}" if row['n_UpA'] > 0 else ""
        cpg = f"CpG×{row['n_CpG']}" if row['n_CpG'] > 0 else ""
        dinucs = ", ".join(filter(None, [upa, cpg])) or "-"
        
        print(f"  {row['motif']}  log2(E)={row['log2_enrichment']:.2f}  [{dinucs}]")
    
    # UpA enrichment summary
    n_upa_in_top10 = sum(1 for _, row in top10.iterrows() if row['n_UpA'] > 0)
    print(f"\nUpA-containing motifs in top 10: {n_upa_in_top10}/10")
    
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
