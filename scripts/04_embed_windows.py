#!/usr/bin/env python3
"""
04_embed_windows.py
===================
Generate DNABERT-2 embeddings for window sequences.

Requires GPU for efficient computation (~2-4 hours on V100 for ~60k windows).

Usage:
    python scripts/04_embed_windows.py \
        --windows data/windows/windows_metadata.csv \
        --fasta data/windows/windows.fasta \
        --output data/embeddings/

Author: [Author Name]
Date: 2026
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_dnabert2(model_name: str = "zhihan1996/DNABERT-2-117M"):
    """
    Load DNABERT-2 model and tokenizer.
    
    Returns
    -------
    tuple
        (tokenizer, model, device)
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    return tokenizer, model, device


def embed_sequence(sequence: str, tokenizer, model, device, 
                   max_length: int = 512) -> np.ndarray:
    """
    Generate embedding for a single sequence using mean pooling.
    
    Parameters
    ----------
    sequence : str
        Nucleotide sequence
    tokenizer : transformers.PreTrainedTokenizer
        DNABERT-2 tokenizer
    model : transformers.PreTrainedModel
        DNABERT-2 model
    device : torch.device
        Computation device
    max_length : int
        Maximum token length (default: 512)
    
    Returns
    -------
    np.ndarray
        768-dimensional embedding vector
    """
    inputs = tokenizer(
        sequence,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embedding = outputs.last_hidden_state.mean(dim=1)
    
    return embedding.cpu().numpy().flatten()


def embed_batch(sequences: list, tokenizer, model, device,
                max_length: int = 512) -> np.ndarray:
    """
    Generate embeddings for a batch of sequences.
    
    Parameters
    ----------
    sequences : list
        List of nucleotide sequences
    tokenizer : transformers.PreTrainedTokenizer
        DNABERT-2 tokenizer
    model : transformers.PreTrainedModel
        DNABERT-2 model
    device : torch.device
        Computation device
    max_length : int
        Maximum token length (default: 512)
    
    Returns
    -------
    np.ndarray
        Array of shape (batch_size, 768)
    """
    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        truncation=True,
        max_length=max_length,
        padding=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Generate DNABERT-2 embeddings for window sequences'
    )
    parser.add_argument(
        '--windows', required=True,
        help='Window metadata CSV file'
    )
    parser.add_argument(
        '--fasta', required=True,
        help='Window sequences FASTA file'
    )
    parser.add_argument(
        '--output', required=True,
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Batch size for embedding (default: 32)'
    )
    parser.add_argument(
        '--model', default="zhihan1996/DNABERT-2-117M",
        help='Model name or path (default: zhihan1996/DNABERT-2-117M)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load window metadata
    print("Loading window metadata...")
    windows_df = pd.read_csv(args.windows)
    print(f"  {len(windows_df)} windows to embed")
    
    # Load sequences
    print("\nLoading sequences...")
    sequences = {}
    for record in SeqIO.parse(args.fasta, 'fasta'):
        sequences[record.id] = str(record.seq).upper()
    print(f"  Loaded {len(sequences)} sequences")
    
    # Verify alignment
    missing = set(windows_df['window_id']) - set(sequences.keys())
    if missing:
        print(f"  Warning: {len(missing)} windows missing sequences")
        windows_df = windows_df[windows_df['window_id'].isin(sequences.keys())]
    
    # Load model
    print("\n" + "="*50)
    tokenizer, model, device = load_dnabert2(args.model)
    print("="*50)
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings (batch_size={args.batch_size})...")
    
    window_ids = windows_df['window_id'].tolist()
    all_embeddings = []
    
    for i in tqdm(range(0, len(window_ids), args.batch_size)):
        batch_ids = window_ids[i:i + args.batch_size]
        batch_seqs = [sequences[wid] for wid in batch_ids]
        
        batch_embeddings = embed_batch(
            batch_seqs, tokenizer, model, device
        )
        all_embeddings.append(batch_embeddings)
        
        # Clear GPU cache periodically
        if device.type == 'cuda' and (i // args.batch_size) % 100 == 0:
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings_array = np.vstack(all_embeddings)
    
    # Save embeddings
    embeddings_file = output_dir / "window_embeddings.npy"
    np.save(embeddings_file, embeddings_array)
    
    # Save window IDs for alignment
    ids_file = output_dir / "window_ids.txt"
    with open(ids_file, 'w') as f:
        for wid in window_ids:
            f.write(f"{wid}\n")
    
    # Print summary
    print("\n" + "="*50)
    print("EMBEDDING SUMMARY")
    print("="*50)
    print(f"Windows embedded:     {embeddings_array.shape[0]:>10,}")
    print(f"Embedding dimension:  {embeddings_array.shape[1]:>10}")
    print(f"Total size:           {embeddings_array.nbytes / 1e6:.1f} MB")
    print("-"*50)
    print(f"\nOutput files:")
    print(f"  Embeddings: {embeddings_file}")
    print(f"  Window IDs: {ids_file}")


if __name__ == '__main__':
    main()
