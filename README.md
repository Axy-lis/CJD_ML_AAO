# CJD ML Pipeline

Code and curated data for the CJD antibody ranking pipeline used in the accompanying paper.

## Repository Layout

- `src/main_project/`: core feature engineering, training, ranking, and validation scripts
- `src/data/`: curated input tables and derived CDR3 datasets used by the pipeline
- `src/features/`: saved feature matrices used for model training
- `src/results/`: ranked output tables
- `src/patent_data/`: patent-derived sequence tables

## Quick Start

1. Create a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the main scoring pipeline from the repository root:

```bash
python src/main_project/ensemble_mutate_rank_prion_traced.py \
  --input src/main_project/positives.csv \
  --neutral_model src/main_project/models/neutralization_like.pkl \
  --select_model src/main_project/models/selectivity_like.pkl \
  --kmer_vectorizer src/main_project/models/kmer_vectorizer_3mer.joblib \
  --out_baseline src/results/baseline_ranked.csv \
  --out_mutants src/results/mutants_ranked.csv \
  --out_lineage src/main_project/figures_ranking/mutation_lineage.csv
```

## Data Notes

The large raw AIRR source files used during preprocessing are intentionally excluded from version control because they exceed GitHub's normal file size limits. The derived public-ready tables needed for scoring and validation remain in this repository.
