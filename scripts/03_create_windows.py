#!/usr/bin/env python3
"""
03_create_windows.py
====================
Generate overlapping windows from genome sequences.

Creates 1000-nt windows with 500-nt stride (50% overlap).

Usage:
    python scripts/03_create_windows.py \
        --fasta data/sequences/genomes_qc.fasta \
        --metadata data/metadata.csv \
        --output data/windows/

Author: [Author Name]
Date: 2026
"""

import argparse
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def create_windows(fasta_path: str, 
                   metadata_df: pd.DataFrame,
                   window_size: int = 1000,
                   stride: int = 500) -> tuple:
    """
    Generate overlapping windows from genome sequences.
    
    Parameters
    ----------
    fasta_path : str
        Path to FASTA file with genome sequences
    metadata_df : pd.DataFrame
        Metadata with 'accession' and 'label' columns
    window_size : int
        Window length in nucleotides (default: 1000)
    stride : int
        Step size between windows (default: 500)
    
    Returns
    -------
    tuple
        (windows_df, sequences_dict)
    """
    # Create accession to metadata lookup
    meta_lookup = metadata_df.set_index('accession').to_dict('index')
    
    windows = []
    sequences = {}
    
    for record in tqdm(SeqIO.parse(fasta_path, 'fasta'), desc="Creating windows"):
        accession = record.id.split('.')[0]
        
        # Skip if not in metadata
        if accession not in meta_lookup:
            continue
        
        meta = meta_lookup[accession]
        seq = str(record.seq).upper()
        genome_length = len(seq)
        
        window_idx = 0
        for start in range(0, genome_length - window_size + 1, stride):
            end = start + window_size
            window_seq = seq[start:end]
            
            # Calculate relative position
            pos_rel = start / genome_length
            pos_bin = min(int(pos_rel * 20), 19)  # 0-indexed, 20 bins
            
            window_id = f"{accession}_{start}_{end}"
            
            windows.append({
                'window_id': window_id,
                'accession': accession,
                'window_idx': window_idx,
                'start': start,
                'end': end,
                'genome_length': genome_length,
                'pos_rel': pos_rel,
                'label': meta.get('label', 'unknown'),
                'species': meta.get('species', 'unknown'),
                'genus': meta.get('genus', 'unknown'),
                'pos_bin': pos_bin
            })
            
            sequences[window_id] = window_seq
            window_idx += 1
    
    windows_df = pd.DataFrame(windows)
    return windows_df, sequences


def main():
    parser = argparse.ArgumentParser(
        description='Generate overlapping windows from genome sequences'
    )
    parser.add_argument(
        '--fasta', required=True,
        help='Input FASTA file with genome sequences'
    )
    parser.add_argument(
        '--metadata', required=True,
        help='Metadata CSV with accession and label columns'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for window files'
    )
    parser.add_argument(
        '--window-size', type=int, default=1000,
        help='Window size in nucleotides (default: 1000)'
    )
    parser.add_argument(
        '--stride', type=int, default=500,
        help='Stride between windows (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(args.metadata)
    print(f"  Loaded {len(metadata_df)} genomes")
    
    # Check for required columns
    if 'label' not in metadata_df.columns:
        raise ValueError("Metadata must contain 'label' column")
    
    # Create windows
    print(f"\nCreating windows (size={args.window_size}, stride={args.stride})...")
    windows_df, sequences = create_windows(
        args.fasta, 
        metadata_df,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Save window metadata
    windows_csv = output_dir / "windows_metadata.csv"
    windows_df.to_csv(windows_csv, index=False)
    
    # Save sequences as FASTA
    windows_fasta = output_dir / "windows.fasta"
    with open(windows_fasta, 'w') as f:
        for window_id, seq in tqdm(sequences.items(), desc="Writing FASTA"):
            f.write(f">{window_id}\n{seq}\n")
    
    # Print summary
    print("\n" + "="*50)
    print("WINDOW CREATION SUMMARY")
    print("="*50)
    print(f"Total windows created:    {len(windows_df):>8,}")
    print(f"Unique genomes:           {windows_df['accession'].nunique():>8,}")
    print(f"Windows per genome:       {len(windows_df) / windows_df['accession'].nunique():.1f}")
    print("-"*50)
    
    # By label
    print("\nBy host range:")
    for label, count in windows_df['label'].value_counts().items():
        print(f"  {label}: {count:,} windows")
    
    print(f"\nOutput files:")
    print(f"  Metadata: {windows_csv}")
    print(f"  Sequences: {windows_fasta}")


if __name__ == '__main__':
    main()
