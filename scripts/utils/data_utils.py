"""
data_utils.py
=============
Data loading and preprocessing utilities.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO


def load_metadata(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load genome metadata CSV file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to metadata CSV
    
    Returns
    -------
    pd.DataFrame
        Metadata with columns: accession, label, species, genus, etc.
    """
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required = ['accession', 'label']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def load_windows(metadata_path: Union[str, Path],
                 embeddings_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Load window metadata and optionally embeddings.
    
    Parameters
    ----------
    metadata_path : str or Path
        Path to windows_metadata.csv
    embeddings_path : str or Path, optional
        Path to window_embeddings.npy
    
    Returns
    -------
    tuple
        (windows_df, embeddings) where embeddings is None if not provided
    """
    windows_df = pd.read_csv(metadata_path)
    
    embeddings = None
    if embeddings_path is not None:
        embeddings = np.load(embeddings_path)
        
        if len(embeddings) != len(windows_df):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) doesn't match "
                f"window count ({len(windows_df)})"
            )
    
    return windows_df, embeddings


def load_embeddings(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load embeddings from NPY file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to embeddings NPY file
    
    Returns
    -------
    np.ndarray
        Embeddings array of shape (n_samples, embedding_dim)
    """
    embeddings = np.load(filepath)
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings


def load_sequences(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load sequences from FASTA file into dictionary.
    
    Parameters
    ----------
    fasta_path : str or Path
        Path to FASTA file
    
    Returns
    -------
    dict
        {sequence_id: sequence_string}
    """
    sequences = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        sequences[record.id] = str(record.seq).upper()
    return sequences


def compute_genome_embeddings(window_embeddings: np.ndarray,
                               windows_df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Compute genome-level embeddings by mean-pooling window embeddings.
    
    Parameters
    ----------
    window_embeddings : np.ndarray
        Window-level embeddings (n_windows, embedding_dim)
    windows_df : pd.DataFrame
        Window metadata with 'accession' column
    
    Returns
    -------
    tuple
        (genome_embeddings, genome_df)
    """
    genome_embeddings = []
    genome_data = []
    
    for accession in windows_df['accession'].unique():
        mask = windows_df['accession'] == accession
        indices = np.where(mask)[0]
        
        # Mean pooling
        genome_emb = window_embeddings[indices].mean(axis=0)
        genome_embeddings.append(genome_emb)
        
        # Get metadata
        row = windows_df[mask].iloc[0]
        genome_data.append({
            'accession': accession,
            'label': row['label'],
            'species': row.get('species', 'unknown'),
            'genus': row.get('genus', 'unknown'),
            'n_windows': len(indices)
        })
    
    genome_embeddings = np.vstack(genome_embeddings)
    genome_df = pd.DataFrame(genome_data)
    
    return genome_embeddings, genome_df


def stratify_by_score(windows_df: pd.DataFrame,
                      score_col: str,
                      top_frac: float = 0.05,
                      bottom_frac: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratify windows into top and bottom groups by prediction score.
    
    Returns
    -------
    tuple
        (top_windows, bottom_windows)
    """
    top_thresh = windows_df[score_col].quantile(1 - top_frac)
    bot_thresh = windows_df[score_col].quantile(bottom_frac)
    
    top_windows = windows_df[windows_df[score_col] >= top_thresh]
    bottom_windows = windows_df[windows_df[score_col] <= bot_thresh]
    
    return top_windows, bottom_windows
