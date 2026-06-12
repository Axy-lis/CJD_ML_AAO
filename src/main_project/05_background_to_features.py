import os
import glob
import json
import re
import argparse

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


def physchem_features(seq: str):
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


def load_background_cdr3s(
    input_path: str, col: str = "cdr3", max_rows: int | None = None
) -> pd.DataFrame:
    """
    input_path can be:
      - a directory containing *.csv
      - a single csv file
    """
    paths = []
    if os.path.isdir(input_path):
        paths = sorted(glob.glob(os.path.join(input_path, "*.csv")))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in directory: {input_path}")
    elif os.path.isfile(input_path):
        paths = [input_path]
    else:
        raise FileNotFoundError(f"Not a file or directory: {input_path}")
    out = []
    remaining = max_rows
    for p in paths:
        df = pd.read_csv(p)
        if col not in df.columns:
            raise ValueError(
                f"{p}: missing column '{col}'. Columns: {list(df.columns)}"
            )
        s = df[col].apply(clean_cdr3)
        s = s[s.str.len() > 0]
        if remaining is not None:
            s = s.iloc[:remaining]
            remaining -= len(s)
            if remaining <= 0:
                remaining = 0
        tmp = pd.DataFrame({"cdr3": s.values, "source_file": os.path.basename(p)})
        out.append(tmp)
        if max_rows is not None and remaining == 0:
            break
    bg = pd.concat(out, ignore_index=True)
    return bg


def main():
    ap = argparse.ArgumentParser(
        description="Convert background CDR3 CSVs into feature vectors using target k-mer vocabulary."
    )
    ap.add_argument(
        "--background",
        required=True,
        help="Path to background directory OR a single CSV file.",
    )
    ap.add_argument(
        "--targets_csvs",
        nargs="+",
        required=True,
        help="The same target CSVs you used earlier (to rebuild vocabulary).",
    )
    ap.add_argument(
        "--meta",
        default="features/targets_v1/feature_meta.json",
        help="Path to feature_meta.json.",
    )
    ap.add_argument(
        "--outdir", default="features/background_v1", help="Output directory."
    )
    ap.add_argument("--col", default="cdr3", help="Column containing CDR3s.")
    ap.add_argument(
        "--max_rows",
        type=int,
        default=50000,
        help="Max background rows to process (default 50k). Increase later.",
    )
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    ngram_min, ngram_max = meta["kmer_vectorizer"]["ngram_range"]
    min_df = meta["kmer_vectorizer"]["min_df"]
    target_rows = []
    for p in args.targets_csvs:
        df = pd.read_csv(p)
        if args.col not in df.columns:
            raise ValueError(
                f"{p}: missing column '{args.col}'. Columns: {list(df.columns)}"
            )
        target_rows.extend([clean_cdr3(x) for x in df[args.col].tolist()])
    target_rows = [x for x in target_rows if len(x) > 0]
    if len(target_rows) == 0:
        raise ValueError(
            "No valid target CDR3 sequences found (cannot rebuild vocabulary)."
        )
    vect = CountVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        lowercase=False,
        min_df=min_df,
    )
    vect.fit(target_rows)
    bg_df = load_background_cdr3s(args.background, col=args.col, max_rows=args.max_rows)
    if len(bg_df) == 0:
        raise ValueError("No valid background CDR3 sequences found after cleaning.")
    X_kmer_bg = vect.transform(bg_df["cdr3"].tolist()).tocsr()
    X_phys_bg = np.array(
        [physchem_features(s) for s in bg_df["cdr3"].tolist()], dtype=np.float32
    )
    rows_path = os.path.join(args.outdir, "rows_bg.csv")
    kmer_path = os.path.join(args.outdir, "X_kmer_bg.npz")
    phys_path = os.path.join(args.outdir, "X_phys_bg.npy")
    bg_df.to_csv(rows_path, index=False)
    sparse.save_npz(kmer_path, X_kmer_bg)
    np.save(phys_path, X_phys_bg)
    print("Saved background features:")
    print(" ", rows_path)
    print(" ", kmer_path)
    print(" ", phys_path)
    print("Shapes:")
    print("  X_kmer_bg:", X_kmer_bg.shape)
    print("  X_phys_bg:", X_phys_bg.shape)
    print("  k-mer vocab size:", len(vect.vocabulary_))
    print("  processed background rows:", len(bg_df))


if __name__ == "__main__":
    main()
