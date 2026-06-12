import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

AA_HYDRO = set("AILMFWV")
AA_AROM = set("FYW")
AA_POS = set("KRH")
AA_NEG = set("DE")
AA_GP = set("GP")

VALID_AA_RE = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")


def clean_cdr3(s: str) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s).strip().upper()
    s = VALID_AA_RE.sub("", s)
    return s


def physchem_features(seq: str) -> List[float]:
    """7 dense features from a CDR3 string."""
    L = len(seq)
    if L == 0:
        return [0.0] * 7
    hyd = sum(a in AA_HYDRO for a in seq) / L
    arom = sum(a in AA_AROM for a in seq) / L
    pos = sum(a in AA_POS for a in seq) / L
    neg = sum(a in AA_NEG for a in seq) / L
    net = (sum(a in AA_POS for a in seq) - sum(a in AA_NEG for a in seq)) / L
    gp = sum(a in AA_GP for a in seq) / L
    return [
        float(L),
        float(hyd),
        float(arom),
        float(pos),
        float(neg),
        float(net),
        float(gp),
    ]


def load_cdr3s(csv_paths: List[str], col: str = "cdr3") -> pd.DataFrame:
    """Load multiple csvs -> one dataframe with columns: cdr3, source_file."""
    rows = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if col not in df.columns:
            raise ValueError(
                f"{p}: missing required column '{col}'. Columns: {list(df.columns)}"
            )
        tmp = pd.DataFrame(
            {
                "cdr3": df[col].apply(clean_cdr3),
                "source_file": os.path.basename(p),
            }
        )
        rows.append(tmp)
    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df[all_df["cdr3"].str.len() > 0].copy()
    return all_df


@dataclass
class FeatureOutputs:
    X_kmer: sparse.csr_matrix
    X_phys: np.ndarray
    vectorizer: CountVectorizer
    rows_df: pd.DataFrame


def make_features(
    rows_df: pd.DataFrame,
    ngram_min: int = 3,
    ngram_max: int = 3,
    min_df: int = 1,
) -> FeatureOutputs:
    seqs = rows_df["cdr3"].tolist()
    vect = CountVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        min_df=min_df,
    )
    X_kmer = vect.fit_transform(seqs).tocsr()
    X_phys = np.array([physchem_features(s) for s in seqs], dtype=np.float32)
    return FeatureOutputs(
        X_kmer=X_kmer, X_phys=X_phys, vectorizer=vect, rows_df=rows_df
    )


def save_outputs(outdir: str, feats: FeatureOutputs):
    os.makedirs(outdir, exist_ok=True)
    rows_path = os.path.join(outdir, "rows.csv")
    feats.rows_df.to_csv(rows_path, index=False)
    kmer_path = os.path.join(outdir, "X_kmer.npz")
    sparse.save_npz(kmer_path, feats.X_kmer)
    phys_path = os.path.join(outdir, "X_phys.npy")
    np.save(phys_path, feats.X_phys)
    meta = {
        "kmer_vectorizer": {
            "ngram_range": list(feats.vectorizer.ngram_range),
            "min_df": feats.vectorizer.min_df,
            "vocab_size": int(len(feats.vectorizer.vocabulary_)),
            "feature_names_preview": feats.vectorizer.get_feature_names_out()[
                :25
            ].tolist(),
        },
        "physchem_feature_names": [
            "length",
            "hydrophobic_fraction",
            "aromatic_fraction",
            "positive_fraction",
            "negative_fraction",
            "net_charge_per_residue",
            "gly_pro_fraction",
        ],
        "shapes": {
            "X_kmer": [int(feats.X_kmer.shape[0]), int(feats.X_kmer.shape[1])],
            "X_phys": [int(feats.X_phys.shape[0]), int(feats.X_phys.shape[1])],
        },
    }
    meta_path = os.path.join(outdir, "feature_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("Saved:")
    print(" ", rows_path)
    print(" ", kmer_path)
    print(" ", phys_path)
    print(" ", meta_path)
    print("Shapes:")
    print("  X_kmer:", feats.X_kmer.shape)
    print("  X_phys:", feats.X_phys.shape)


def main():
    ap = argparse.ArgumentParser(
        description="Convert CDR3 CSVs into feature vectors (k-mers + physchem)."
    )
    ap.add_argument("--csvs", nargs="+", required=True, help="One or more CSV paths.")
    ap.add_argument(
        "--col",
        default="cdr3",
        help="Column name containing CDR3 sequences (default: cdr3).",
    )
    ap.add_argument("--outdir", default="features_out", help="Output directory.")
    ap.add_argument(
        "--ngram_min", type=int, default=3, help="Min k-mer size (default: 3)."
    )
    ap.add_argument(
        "--ngram_max", type=int, default=3, help="Max k-mer size (default: 3)."
    )
    ap.add_argument(
        "--min_df", type=int, default=1, help="Drop k-mers seen in < min_df sequences."
    )
    args = ap.parse_args()
    rows_df = load_cdr3s(args.csvs, col=args.col)
    if len(rows_df) == 0:
        raise ValueError("No valid CDR3 sequences found after cleaning.")
    feats = make_features(
        rows_df,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
    )
    save_outputs(args.outdir, feats)


if __name__ == "__main__":
    main()
