#!/usr/bin/env python3
"""
01_fetch_sequences.py
=====================
Download flavivirus genome sequences from NCBI GenBank.

Usage:
    python scripts/01_fetch_sequences.py \
        --arb data/accessions/arb_accessions.txt \
        --isv data/accessions/isv_accessions.txt \
        --output data/sequences/

Author: [Author Name]
Date: 2026
"""

import argparse
import os
import time
from pathlib import Path

from Bio import Entrez, SeqIO
from tqdm import tqdm


def load_accessions(filepath: str) -> list:
    """Load accession numbers from a text file (one per line)."""
    with open(filepath, 'r') as f:
        accessions = [line.strip() for line in f if line.strip()]
    return accessions


def fetch_sequences(accessions: list, output_fasta: str, 
                    batch_size: int = 100, email: str = None) -> int:
    """
    Fetch sequences from NCBI in batches.
    
    Parameters
    ----------
    accessions : list
        List of NCBI accession numbers
    output_fasta : str
        Output FASTA file path
    batch_size : int
        Number of sequences to fetch per request (default: 100)
    email : str
        Email address for NCBI Entrez (required by NCBI)
    
    Returns
    -------
    int
        Number of sequences successfully fetched
    """
    if email:
        Entrez.email = email
    else:
        # NCBI E-utilities require a contact email. Pass email=... or set the
        # NCBI_EMAIL environment variable before running.
        env_email = os.environ.get("NCBI_EMAIL")
        if not env_email:
            raise ValueError(
                "NCBI requires a contact email. Pass email='you@example.org' "
                "or set the NCBI_EMAIL environment variable."
            )
        Entrez.email = env_email
    
    fetched_count = 0
    
    with open(output_fasta, 'w') as out_handle:
        for i in tqdm(range(0, len(accessions), batch_size), 
                      desc="Fetching batches"):
            batch = accessions[i:i + batch_size]
            
            try:
                handle = Entrez.efetch(
                    db='nucleotide',
                    id=batch,
                    rettype='fasta',
                    retmode='text'
                )
                
                fasta_data = handle.read()
                out_handle.write(fasta_data)
                handle.close()
                
                # Count sequences in this batch
                fetched_count += fasta_data.count('>')
                
            except Exception as e:
                print(f"Error fetching batch {i//batch_size + 1}: {e}")
                # Continue with next batch
            
            # Be nice to NCBI servers
            time.sleep(0.5)
    
    return fetched_count


def main():
    parser = argparse.ArgumentParser(
        description='Fetch flavivirus genomes from NCBI GenBank'
    )
    parser.add_argument(
        '--arb', required=True,
        help='Path to arbovirus accession list'
    )
    parser.add_argument(
        '--isv', required=True,
        help='Path to ISV accession list'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for FASTA files'
    )
    parser.add_argument(
        '--email', default=None,
        help='Email address for NCBI Entrez (required by NCBI)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=100,
        help='Batch size for NCBI requests (default: 100)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load accession lists
    print("Loading accession lists...")
    arb_accessions = load_accessions(args.arb)
    isv_accessions = load_accessions(args.isv)
    
    print(f"  ARB accessions: {len(arb_accessions)}")
    print(f"  ISV accessions: {len(isv_accessions)}")
    
    # Fetch ARB sequences
    print("\nFetching arbovirus sequences...")
    arb_output = output_dir / "arb_genomes.fasta"
    arb_count = fetch_sequences(
        arb_accessions, str(arb_output), 
        batch_size=args.batch_size, email=args.email
    )
    print(f"  Fetched {arb_count} ARB sequences")
    
    # Fetch ISV sequences
    print("\nFetching ISV sequences...")
    isv_output = output_dir / "isv_genomes.fasta"
    isv_count = fetch_sequences(
        isv_accessions, str(isv_output),
        batch_size=args.batch_size, email=args.email
    )
    print(f"  Fetched {isv_count} ISV sequences")
    
    # Combined file
    print("\nCreating combined FASTA...")
    combined_output = output_dir / "all_genomes.fasta"
    with open(combined_output, 'w') as out:
        for fasta_file in [arb_output, isv_output]:
            with open(fasta_file, 'r') as f:
                out.write(f.read())
    
    print(f"\nDone! Total sequences: {arb_count + isv_count}")
    print(f"Output files:")
    print(f"  {arb_output}")
    print(f"  {isv_output}")
    print(f"  {combined_output}")


if __name__ == '__main__':
    main()
