#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import joblib


def load_features(kmer_npz: str, phys_npy: str):
    X_kmer = load_npz(kmer_npz).tocsr()
    X_phys = np.load(phys_npy)
    if X_phys.ndim != 2:
        raise ValueError(f"Expected 2D phys array, got shape {X_phys.shape}")
    if X_kmer.shape[0] != X_phys.shape[0]:
        raise ValueError(
            f"Row mismatch: kmer has {X_kmer.shape[0]} rows, phys has {X_phys.shape[0]} rows"
        )
    return X_kmer, X_phys


def concat_kmer_phys(
    X_kmer: sp.csr_matrix, X_phys: np.ndarray, phys_scaler: StandardScaler
):
    X_phys_scaled = phys_scaler.transform(X_phys)
    X_phys_sp = sp.csr_matrix(X_phys_scaled)
    X = sp.hstack([X_kmer, X_phys_sp], format="csr")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--targets_kmer", required=True, help="e.g., features/targets_v1/X_kmer.npz"
    )
    ap.add_argument(
        "--targets_phys", required=True, help="e.g., features/targets_v1/X_phys.npy"
    )
    ap.add_argument(
        "--background_kmer",
        required=True,
        help="e.g., features/background_v1/X_kmer_bg.npz",
    )
    ap.add_argument(
        "--background_phys",
        required=True,
        help="e.g., features/background_v1/X_phys_bg.npy",
    )
    ap.add_argument(
        "--targets_rows",
        help="CSV mapping targets rows to source_file (must match targets feature row order)",
    )
    ap.add_argument(
        "--pos_source",
        help="Which source_file value to treat as positive, e.g. neutralization_criteria.csv",
    )
    ap.add_argument(
        "--other_targets_as_negative",
        action="store_true",
        help="If set, other target rows become negatives (one-vs-rest). If not set, other target rows are dropped.",
    )
    ap.add_argument(
        "--out_model", required=True, help="e.g., models/neutralization_like.pkl"
    )
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C", type=float, default=1.0)
    args = ap.parse_args()
    Xt_kmer, Xt_phys = load_features(args.targets_kmer, args.targets_phys)
    Xb_kmer, Xb_phys = load_features(args.background_kmer, args.background_phys)
    if (args.targets_rows is None) ^ (args.pos_source is None):
        raise ValueError(
            "Provide BOTH --targets_rows and --pos_source, or provide NEITHER (legacy all-targets-positive mode)."
        )
    if args.targets_rows is not None:
        rows_t = pd.read_csv(args.targets_rows)
        if "source_file" not in rows_t.columns:
            raise ValueError("targets_rows must contain column 'source_file'")
        if len(rows_t) != Xt_kmer.shape[0]:
            raise ValueError(
                f"targets_rows row count {len(rows_t)} != targets features rows {Xt_kmer.shape[0]}"
            )
        pos_mask = rows_t["source_file"].astype(str) == args.pos_source
        if args.other_targets_as_negative:
            yt = pos_mask.astype(np.int32).values
        else:
            keep = pos_mask.values
            Xt_kmer = Xt_kmer[keep]
            Xt_phys = Xt_phys[keep]
            yt = np.ones(Xt_kmer.shape[0], dtype=np.int32)
    else:
        yt = np.ones(Xt_kmer.shape[0], dtype=np.int32)
    yb = np.zeros(Xb_kmer.shape[0], dtype=np.int32)
    X_kmer_all = sp.vstack([Xt_kmer, Xb_kmer], format="csr")
    X_phys_all = np.vstack([Xt_phys, Xb_phys])
    y_all = np.concatenate([yt, yb])
    n_pos = int(y_all.sum())
    n_neg = int((y_all == 0).sum())
    print(f"Training labels: pos={n_pos} neg={n_neg}")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    aucs, aps = [], []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(np.zeros_like(y_all), y_all), start=1
    ):
        Xk_tr = X_kmer_all[train_idx]
        Xp_tr = X_phys_all[train_idx]
        y_tr = y_all[train_idx]
        Xk_te = X_kmer_all[test_idx]
        Xp_te = X_phys_all[test_idx]
        y_te = y_all[test_idx]
        scaler = StandardScaler()
        scaler.fit(Xp_tr)
        X_tr = concat_kmer_phys(Xk_tr, Xp_tr, scaler)
        X_te = concat_kmer_phys(Xk_te, Xp_te, scaler)
        clf = LogisticRegression(
            C=args.C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
        )
        clf.fit(X_tr, y_tr)
        p = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, p)
        ap_score = average_precision_score(y_te, p)
        aucs.append(auc)
        aps.append(ap_score)
        print(
            f"[fold {fold}] ROC AUC={auc:.4f} | Avg Precision={ap_score:.4f} | n_test={len(test_idx)}"
        )
    print("\nCV summary")
    print(f"ROC AUC:        mean={np.mean(aucs):.4f}  std={np.std(aucs):.4f}")
    print(f"Avg Precision:  mean={np.mean(aps):.4f}  std={np.std(aps):.4f}")
    final_scaler = StandardScaler()
    final_scaler.fit(X_phys_all)
    X_all = concat_kmer_phys(X_kmer_all, X_phys_all, final_scaler)
    final_clf = LogisticRegression(
        C=args.C,
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=args.seed,
    )
    final_clf.fit(X_all, y_all)
    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_clf,
        "phys_scaler": final_scaler,
        "n_kmer_features": X_kmer_all.shape[1],
        "n_phys_features": X_phys_all.shape[1],
        "C": args.C,
        "pos_source": args.pos_source,
        "other_targets_as_negative": bool(args.other_targets_as_negative),
    }
    joblib.dump(payload, out_path)
    print(f"\nSaved model bundle -> {out_path}")


if __name__ == "__main__":
    main()
