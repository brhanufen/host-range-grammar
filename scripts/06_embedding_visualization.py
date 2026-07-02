#!/usr/bin/env python3
"""
06_embedding_visualization.py
=============================
Visualize embedding space with UMAP/t-SNE and compute clustering metrics.

Usage:
    python scripts/06_embedding_visualization.py \
        --embeddings data/embeddings/window_embeddings.npy \
        --windows data/windows/windows_metadata.csv \
        --output results/

Author: [Author Name]
Date: 2026
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import umap
from tqdm import tqdm


def compute_genome_embeddings(embeddings: np.ndarray, 
                               windows_df: pd.DataFrame) -> tuple:
    """
    Compute genome-level embeddings by averaging window embeddings.
    
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
        genome_emb = embeddings[indices].mean(axis=0)
        genome_embeddings.append(genome_emb)
        
        # Get metadata (same for all windows of this genome)
        row = windows_df[mask].iloc[0]
        genome_data.append({
            'accession': accession,
            'label': row['label'],
            'species': row['species'],
            'genus': row['genus']
        })
    
    genome_embeddings = np.vstack(genome_embeddings)
    genome_df = pd.DataFrame(genome_data)
    
    return genome_embeddings, genome_df


def compute_knn_purity(embeddings: np.ndarray, 
                       labels: np.ndarray,
                       k: int = 15,
                       n_permutations: int = 1000) -> dict:
    """
    Compute kNN purity with permutation-based null distribution.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Feature matrix (n_samples, n_features)
    labels : np.ndarray
        Class labels
    k : int
        Number of nearest neighbors (default: 15)
    n_permutations : int
        Number of permutations for null distribution (default: 1000)
    
    Returns
    -------
    dict
        Dictionary with purity, null mean/std, z-score, p-value
    """
    # Fit kNN
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    
    # Exclude self (first neighbor)
    neighbor_indices = indices[:, 1:]
    
    # Compute observed purity
    neighbor_labels = labels[neighbor_indices]
    purity_per_sample = (neighbor_labels == labels[:, None]).mean(axis=1)
    observed_purity = purity_per_sample.mean()
    
    # Permutation null distribution
    null_purities = []
    for _ in tqdm(range(n_permutations), desc="Permutation test"):
        perm_labels = np.random.permutation(labels)
        perm_neighbor_labels = perm_labels[neighbor_indices]
        perm_purity = (perm_neighbor_labels == perm_labels[:, None]).mean()
        null_purities.append(perm_purity)
    
    null_mean = np.mean(null_purities)
    null_std = np.std(null_purities)
    
    # Z-score and p-value
    z_score = (observed_purity - null_mean) / null_std if null_std > 0 else np.inf
    p_value = np.mean(np.array(null_purities) >= observed_purity)
    
    return {
        'purity': observed_purity,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': max(p_value, 1/n_permutations),  # Floor at 1/n_perm
        'null_distribution': null_purities
    }


def main():
    parser = argparse.ArgumentParser(
        description='Embedding visualization and clustering analysis'
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
        '--k', type=int, default=15,
        help='k for kNN purity (default: 15)'
    )
    parser.add_argument(
        '--n-permutations', type=int, default=1000,
        help='Number of permutations for null distribution (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    embeddings = np.load(args.embeddings)
    windows_df = pd.read_csv(args.windows)
    
    # Compute genome-level embeddings
    print("\nComputing genome-level embeddings...")
    genome_embeddings, genome_df = compute_genome_embeddings(embeddings, windows_df)
    print(f"  {len(genome_df)} genomes")
    
    # UMAP
    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        random_state=42,
        n_components=2
    )
    umap_coords = reducer.fit_transform(genome_embeddings)
    genome_df['umap_x'] = umap_coords[:, 0]
    genome_df['umap_y'] = umap_coords[:, 1]
    
    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2, 
        perplexity=30, 
        random_state=42,
        n_iter=1000
    )
    tsne_coords = tsne.fit_transform(genome_embeddings)
    genome_df['tsne_x'] = tsne_coords[:, 0]
    genome_df['tsne_y'] = tsne_coords[:, 1]
    
    # Prepare binary labels
    labels = (genome_df['label'] == 'ISV').astype(int).values
    
    # kNN purity with permutation test
    print(f"\nComputing kNN purity (k={args.k})...")
    purity_results = compute_knn_purity(
        genome_embeddings, labels, 
        k=args.k, 
        n_permutations=args.n_permutations
    )
    
    # Silhouette score
    print("Computing silhouette score...")
    silhouette = silhouette_score(genome_embeddings, labels)
    
    # Save results
    genome_df.to_csv(output_dir / "genome_embeddings.csv", index=False)
    
    # Save genome embeddings array
    np.save(output_dir / "genome_embeddings.npy", genome_embeddings)
    
    # Save clustering metrics
    metrics_df = pd.DataFrame([{
        'knn_purity': purity_results['purity'],
        'null_mean': purity_results['null_mean'],
        'null_std': purity_results['null_std'],
        'z_score': purity_results['z_score'],
        'p_value': purity_results['p_value'],
        'silhouette_score': silhouette,
        'k': args.k,
        'n_permutations': args.n_permutations
    }])
    metrics_df.to_csv(output_dir / "knn_purity_results.csv", index=False)
    
    # Save null distribution for plotting
    null_df = pd.DataFrame({'null_purity': purity_results['null_distribution']})
    null_df.to_csv(output_dir / "knn_null_distribution.csv", index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS SUMMARY")
    print("="*60)
    print(f"kNN Purity (k={args.k}):    {purity_results['purity']:.4f}")
    print(f"Null distribution:         {purity_results['null_mean']:.4f} ± {purity_results['null_std']:.4f}")
    print(f"Z-score:                   {purity_results['z_score']:.2f}")
    print(f"P-value:                   {purity_results['p_value']:.2e}")
    print(f"Silhouette score:          {silhouette:.4f}")
    print("="*60)
    
    print(f"\nOutput files saved to: {output_dir}")


if __name__ == '__main__':
    main()
