from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DEFAULT_FEATURE_DIR = Path(__file__).resolve().parents[1] / "features" / "targets_v1"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "models" / "kmer_vectorizer_3mer.joblib"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the saved 3-mer vectorizer from target feature rows."
    )
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = pd.read_csv(args.feature_dir / "rows.csv")
    seqs = rows["cdr3"].astype(str).str.upper().tolist()
    vect = CountVectorizer(
        analyzer="char", ngram_range=(3, 3), lowercase=False, min_df=1
    )
    vect.fit(seqs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vect, args.out)
    print("Saved:", args.out)
    print("vocab_size:", len(vect.vocabulary_))


if __name__ == "__main__":
    main()
