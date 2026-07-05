# Host-Range Grammar of Orthoflaviviruses — Reproducibility Package

Reproducibility package for:

> **Deciphering the Host-Range Grammar of Orthoflaviviruses Using Foundation Model Embeddings: A Leakage-Aware Evaluation Framework**
>
> Brhanu F. Znabu, Qiuming Yao, and Nicole R. Sexton
>
> Nebraska Center for Virology, Biomedical Engineering Program, School of Biological Sciences, and School of Computing, University of Nebraska–Lincoln

## Overview

This archive contains all data, code, and pre-computed results needed to reproduce
every result, table, and figure in the manuscript. The study demonstrates that
evaluation leakage inflates the reported performance of machine-learning viral
host-range predictors regardless of feature representation, and identifies candidate
sequence-grammar elements (UpA dinucleotide motifs) linked to orthoflavivirus
host-range determination.

All headline numbers reproduce exactly from the committed inputs; an automated
checker (`verify_reproducibility.py`) confirms 21/21 key values.

## Repository structure

```
.
├── data/
│   ├── metadata.csv                      # 3,031 genome metadata
│   ├── windows_metadata.csv              # 59,980 window metadata
│   ├── windows_sequences.csv             # 59,980 window sequences
│   ├── windows_predictions_FIXED_SEED.csv# Per-window P(ARB) — drives Figure 4
│   └── accessions/
│       ├── arb_accessions.txt            # 2,816 ARB accessions
│       └── isv_accessions.txt            # 215 ISV accessions
├── embeddings/
│   ├── embeddings_combined.npy           # 59,980 × 768 DNABERT-2 embeddings
│   └── embeddings_index.csv              # Window-to-embedding index
├── results/
│   ├── all_baseline_results.json         # AP/SD for all methods & schemes (Fig 2, 6B)
│   ├── ablation_results_repro.json       # Window-masking ablation (Fig 6A)
│   ├── motif_enrichment_full.csv         # 4,095 6-mer enrichment results (Fig 5, Table 1)
│   ├── oof_predictions.npz               # Out-of-fold predictions (Fig 2)
│   ├── figure4_final_repro.json          # Expected Figure 4 values (reference)
│   ├── motif_scorer_results_repro.json   # Transparent motif-scorer CV output
│   └── hotspot_stability_repro.json      # Positional-hotspot stability check
├── scripts/
│   ├── 01_fetch_sequences.py             # Download genomes from NCBI
│   ├── 02_qc_filter.py                   # Quality-control filtering
│   ├── 03_create_windows.py             # Create overlapping 1-kb windows
│   ├── 04_embed_windows.py               # Generate DNABERT-2 embeddings (GPU)
│   ├── 05_grouped_cv.py                  # Leakage-aware cross-validation
│   ├── 06_embedding_visualization.py     # Embedding-space metrics
│   ├── 07_positional_attribution.py      # Genome-localized attribution
│   ├── 08_motif_enrichment.py            # 6-mer motif enrichment
│   ├── 09_ablation_studies.py            # Ablation / robustness
│   ├── 10_baseline_comparison.py         # Baseline-method comparison
│   ├── 11_motif_scorer_cv.py             # Transparent motif-scorer baseline
│   ├── 12_ablation_repro.py              # Ablation reproduction
│   ├── 16_species_group_sensitivity.py  # Species-group sensitivity (Unknown flavivirus)
│   ├── utils/                            # Shared data/eval helpers
│   └── figures/                          # Publication-figure regenerators
│       ├── regen_fig2.py                 # Figure 2
│       ├── regen_fig3.py                 # Figure 3
│       ├── regen_fig4_repro.py           # Figure 4
│       ├── regen_fig5_fig6.py            # Figures 5 and 6
│       ├── regen_fig7.py                 # Figure 7
│       └── regen_figS1.py                # Supplementary Figure S1 + Table S2
├── figures/                              # Figure1–Figure7 (.png and .pdf)
├── supplementary/                        # Supplementary Figure S1, Table S2
├── figures_tiff/                         # 600 DPI TIFFs (Figure1–7)
├── verify_reproducibility.py             # Automated verification (21 checks)
├── requirements.txt
└── README.md
```

*Figure 1 is a schematic (dataset-curation / evaluation-framework diagram) and is
provided as a static image; it is not computationally generated.*

## Dataset summary

| Metric | Value |
|--------|-------|
| Total genomes | 3,031 |
| ARB genomes | 2,816 (23 species) |
| ISFV genomes | 215 (22 species) |
| Total windows | 59,980 |
| Window size / stride | 1,000 bp / 500 bp (50% overlap) |
| Embedding dimensions | 768 (DNABERT-2) |

## Large data files

Two files are large and are provided in full in the permanent Zenodo archive:

| File | Size | Path |
|------|------|------|
| `embeddings_combined.npy` | 176 MB | `embeddings/embeddings_combined.npy` |
| `windows_sequences.csv` | 58 MB | `data/windows_sequences.csv` |

This Zenodo deposit is the complete, self-contained archive: all files above are
included here, so `verify_reproducibility.py` and the full pipeline run directly
after unpacking. (The companion GitHub repository omits these two files for size
reasons and links back to this archive to obtain them.)

## Quick start

```bash
pip install -r requirements.txt
python verify_reproducibility.py        # 21/21 checks should pass
```

`verify_reproducibility.py` retrains the leakage-aware models from the committed
embeddings and checks all key manuscript values (dataset counts, AP values, kNN
purity, silhouette, baselines, motif enrichment) — reporting pass/fail for each.

## Reproducing the figures

All figure scripts read only the committed data/results and use paths relative to
the script, so they run from anywhere after `pip install -r requirements.txt`:

```bash
python scripts/figures/regen_fig2.py        # Figure 2  (evaluation leakage)
python scripts/figures/regen_fig3.py        # Figure 3  (UMAP/t-SNE; SEED=42)
python scripts/figures/regen_fig4_repro.py  # Figure 4  (positional hotspots)
python scripts/figures/regen_fig5_fig6.py   # Figures 5 & 6 (motifs; ablation)
python scripts/figures/regen_fig7.py        # Figure 7  (mechanistic synthesis)
python scripts/figures/regen_figS1.py       # Suppl. Fig S1 (composition features)
```

Each writes `.png` and `.pdf` into `figures/`.

| Figure | Generator | Reproduces |
|--------|-----------|------------|
| 2 | `scripts/figures/regen_fig2.py` | Random 0.984 / Accession 0.981 / Species 0.819; baselines |
| 3 | `scripts/figures/regen_fig3.py` | kNN purity 0.997, silhouette 0.433, null 0.868±0.003, P<0.001 |
| 4 | `scripts/figures/regen_fig4_repro.py` | ARB 1.8×@0.38, ISFV 2.5×@0.57; ARB 47%/65%, ISFV 35%/33% |
| 5 | `scripts/figures/regen_fig5_fig6.py` | AUUAGG log₂E=3.22; UpA-only 37% / CpG-only 66% by decile |
| 6 | `scripts/figures/regen_fig5_fig6.py` | ablation \|ΔAP\|<0.01; motif-scorer 0.638/0.434 |
| 7 | `scripts/figures/regen_fig7.py` | Top-10 motif table (= Table 1) |

## Reproducing from scratch (optional)

The numbered pipeline regenerates the intermediate results. Steps 1–4 require
network access (NCBI) and a GPU with `transformers`/`torch`; **pre-computed
embeddings are provided**, so reviewers can skip to step 5.

```bash
python scripts/01_fetch_sequences.py     # NCBI download
python scripts/02_qc_filter.py
python scripts/03_create_windows.py
python scripts/04_embed_windows.py       # GPU; or use provided embeddings/
python scripts/05_grouped_cv.py
python scripts/06_embedding_visualization.py
python scripts/07_positional_attribution.py
python scripts/08_motif_enrichment.py
python scripts/09_ablation_studies.py
python scripts/10_baseline_comparison.py
python scripts/11_motif_scorer_cv.py
python scripts/12_ablation_repro.py
```

## Key results

### Evaluation leakage is method-agnostic

| Method | Accession-grouped AP | Species-grouped AP | Drop (pp) |
|--------|---------------------:|-------------------:|----------:|
| DNABERT-2 | 0.981 ± 0.003 | 0.819 ± 0.146 | 16 |
| k-mer TF-IDF | 0.997 ± 0.004 | 0.858 ± 0.183 | 14 |
| Composition | 0.928 ± 0.010 | 0.834 ± 0.150 | 9 |
| Position-only | 0.071 ± 0.006 | 0.070 ± 0.049 | ~0 |

### Embedding space

| Metric | Value |
|--------|-------|
| kNN purity (k=15) | 0.997 |
| Silhouette score | 0.433 |
| Permutation null | 0.868 ± 0.003 (P < 0.001) |

### Motif enrichment

| Finding | Value |
|---------|-------|
| Top discriminative motif | AUUAGG (log₂E = 3.22, 9.3-fold) |
| UpA-only motifs in top 10 | 6 / 10 |
| UpA-only in most ARB-enriched decile | 37% (vs 24% expected) |
| CpG-only in most ISFV-enriched decile | 66% (vs 24% expected) |
| Mann–Whitney P (UpA vs CpG) | < 10⁻¹²⁶ |
| 6-mers retained | 4,095 (of 4,096 possible; 1 below min-prevalence) |

### Positional attribution (Figure 4)

| Class | Other (5′UTR–NS2B) | NS3–NS5 |
|-------|-------------------:|--------:|
| ARB | 47% | 65% |
| ISFV | 35% | 33% |

## Repository contents

- `figures/` — Figures 1–7 (PDF + PNG, 600 DPI)
- `scripts/` — full analysis pipeline (01–16) plus figure generators (`scripts/figures/`) and utilities (`scripts/utils/`)
- `data/` — accession lists, metadata, window sequences and predictions
- `embeddings/` — DNABERT-2 genome-mean embeddings (`embeddings_combined.npy`) and index
- `results/` — reproducible result files (JSON/CSV/NPZ) for the ablation, baseline, motif, and attribution analyses
- `requirements.txt`, `verify_reproducibility.py` — environment and reproducibility checks

## Note on figure revision

Figures 1–7 and their generator scripts (`scripts/figures/regen_fig*.py`) reflect a
consistency pass applied prior to deposit: the insect-specific-flavivirus (ISFV) class
colour was harmonized to a single orange (`#E6550D`) across all figures, Figure 1A merges
the window-count label into the pipeline box, Figure 3A/3B place the class legend below the
panels, panel letters are unified, and thin strokes were set to a >=0.5 pt floor. No
underlying data or numeric values changed; all reproducibility outputs are unaffected.

## Citation

If you use this archive, please cite the Zenodo deposit:

> Znabu, B. F., Yao, Q., & Sexton, N. R. (2026). Deciphering the Host-Range Grammar of Orthoflaviviruses Using Foundation Model Embeddings: A Leakage-Aware Evaluation Framework — Data and Code. Zenodo. https://doi.org/10.5281/zenodo.21209701

- Concept DOI (all versions): https://doi.org/10.5281/zenodo.21209701
- This version: https://doi.org/10.5281/zenodo.21209702

The associated manuscript citation will be added upon publication.

## Contact

Nicole R. Sexton (Corresponding Author) — Nebraska Center for Virology,
School of Biological Sciences, University of Nebraska–Lincoln.

Qiuming Yao — Nebraska Center for Virology and School of Computing, University of Nebraska–Lincoln.
