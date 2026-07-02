#!/usr/bin/env python3
"""
02_qc_filter.py
===============
Quality control filtering for flavivirus genomes.

Filters:
1. Genome length: 9,000-12,000 nt (flavivirus range)
2. Ambiguous bases: <1% N content
3. No exact duplicates (by sequence hash)

Usage:
    python scripts/02_qc_filter.py \
        --input data/sequences/all_genomes.fasta \
        --output data/sequences/genomes_qc.fasta \
        --metadata data/metadata.csv

Author: [Author Name]
Date: 2026
"""

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from tqdm import tqdm


def compute_seq_hash(sequence: str) -> str:
    """Compute MD5 hash of a sequence for duplicate detection."""
    return hashlib.md5(sequence.upper().encode()).hexdigest()


def compute_gc_content(sequence: str) -> float:
    """Compute GC content of a sequence."""
    seq = sequence.upper()
    gc = seq.count('G') + seq.count('C')
    total = len(seq) - seq.count('N')
    return gc / total if total > 0 else 0


def compute_n_fraction(sequence: str) -> float:
    """Compute fraction of ambiguous bases (N)."""
    seq = sequence.upper()
    return seq.count('N') / len(seq) if len(seq) > 0 else 0


def filter_genomes(input_fasta: str, 
                   min_length: int = 9000,
                   max_length: int = 12000,
                   max_n_fraction: float = 0.01) -> tuple:
    """
    Apply QC filters to genome sequences.
    
    Parameters
    ----------
    input_fasta : str
        Path to input FASTA file
    min_length : int
        Minimum genome length (default: 9000)
    max_length : int
        Maximum genome length (default: 12000)
    max_n_fraction : float
        Maximum fraction of N bases (default: 0.01 = 1%)
    
    Returns
    -------
    tuple
        (passed_records, metadata_records, filter_stats)
    """
    seen_hashes = set()
    passed_records = []
    metadata_records = []
    
    filter_stats = defaultdict(int)
    filter_stats['total'] = 0
    
    for record in tqdm(SeqIO.parse(input_fasta, 'fasta'), desc="Filtering"):
        filter_stats['total'] += 1
        
        seq = str(record.seq).upper()
        length = len(seq)
        n_frac = compute_n_fraction(seq)
        gc = compute_gc_content(seq)
        seq_hash = compute_seq_hash(seq)
        
        # Apply filters
        if length < min_length:
            filter_stats['too_short'] += 1
            continue
        
        if length > max_length:
            filter_stats['too_long'] += 1
            continue
        
        if n_frac >= max_n_fraction:
            filter_stats['high_n_content'] += 1
            continue
        
        if seq_hash in seen_hashes:
            filter_stats['duplicate'] += 1
            continue
        
        # Passed all filters
        seen_hashes.add(seq_hash)
        passed_records.append(record)
        
        # Extract accession (handle different ID formats)
        accession = record.id.split('.')[0]
        
        metadata_records.append({
            'accession': accession,
            'full_id': record.id,
            'description': record.description,
            'length': length,
            'gc_content': gc,
            'n_fraction': n_frac
        })
        
        filter_stats['passed'] += 1
    
    return passed_records, metadata_records, dict(filter_stats)


def main():
    parser = argparse.ArgumentParser(
        description='Quality control filtering for flavivirus genomes'
    )
    parser.add_argument(
        '--input', required=True,
        help='Input FASTA file'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output filtered FASTA file'
    )
    parser.add_argument(
        '--metadata', required=True,
        help='Output metadata CSV file'
    )
    parser.add_argument(
        '--min-length', type=int, default=9000,
        help='Minimum genome length (default: 9000)'
    )
    parser.add_argument(
        '--max-length', type=int, default=12000,
        help='Maximum genome length (default: 12000)'
    )
    parser.add_argument(
        '--max-n-fraction', type=float, default=0.01,
        help='Maximum N fraction (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Create output directories
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metadata).parent.mkdir(parents=True, exist_ok=True)
    
    # Run filtering
    print("Applying QC filters...")
    print(f"  Length range: {args.min_length}-{args.max_length} nt")
    print(f"  Max N fraction: {args.max_n_fraction}")
    
    passed, metadata, stats = filter_genomes(
        args.input,
        min_length=args.min_length,
        max_length=args.max_length,
        max_n_fraction=args.max_n_fraction
    )
    
    # Write filtered FASTA
    print(f"\nWriting {len(passed)} filtered sequences...")
    SeqIO.write(passed, args.output, 'fasta')
    
    # Write metadata
    df = pd.DataFrame(metadata)
    df.to_csv(args.metadata, index=False)
    
    # Print summary
    print("\n" + "="*50)
    print("FILTER SUMMARY")
    print("="*50)
    print(f"Total input sequences:    {stats['total']:>8,}")
    print(f"Too short (<{args.min_length}):       {stats.get('too_short', 0):>8,}")
    print(f"Too long (>{args.max_length}):       {stats.get('too_long', 0):>8,}")
    print(f"High N content (≥{args.max_n_fraction}):    {stats.get('high_n_content', 0):>8,}")
    print(f"Duplicates removed:       {stats.get('duplicate', 0):>8,}")
    print("-"*50)
    print(f"Passed QC:                {stats['passed']:>8,}")
    print("="*50)
    
    print(f"\nOutput files:")
    print(f"  Sequences: {args.output}")
    print(f"  Metadata:  {args.metadata}")


if __name__ == '__main__':
    main()
