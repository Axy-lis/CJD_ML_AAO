#!/usr/bin/env python3
"""
ensemble_mutate_rank_prion_traced.py

PRION-CORRECT pipeline (traceable mutation version):
- 3 scorers:
  (1) neutralization_like.pkl  (bundle saved by your training script)
  (2) selectivity_like.pkl     (bundle saved by your training script)
  (3) BBBProxyModel (deterministic proxy from bbb_proxy.py)

Adds:
- mutation lineage tracking parent -> child
- explicit amino-acid replacement logging (pos, from, to, chem group)
- outputs mutation_lineage.csv for poster visuals

Assumes your trained models were fit on:
- CountVectorizer char 3-mers (fixed vocab)
- 7 physchem features in this exact order:
  [length, hydrophobic_fraction, aromatic_fraction, positive_fraction,
   negative_fraction, net_charge_per_residue, gly_pro_fraction]

Outputs:
- baseline_ranked.csv
- mutants_ranked.csv
- mutation_lineage.csv
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib

from bbb_proxy import BBBProxyModel, BBBProxyConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)


def safe_seq(seq: str) -> str:
    seq = (seq or "").strip().upper()
    return "".join([c for c in seq if c in AA_SET])


AA_HYDRO = set("AILMFWV")
AA_AROM = set("FYW")
AA_POS = set("KRH")
AA_NEG = set("DE")
AA_GP = set("GP")


def phys_features_7(seq: str) -> np.ndarray:
    seq = safe_seq(seq)
    L = len(seq)
    if L == 0:
        return np.zeros(7, dtype=np.float64)
    hyd = sum(a in AA_HYDRO for a in seq) / L
    arom = sum(a in AA_AROM for a in seq) / L
    pos = sum(a in AA_POS for a in seq) / L
    neg = sum(a in AA_NEG for a in seq) / L
    net = (sum(a in AA_POS for a in seq) - sum(a in AA_NEG for a in seq)) / L
    gp = sum(a in AA_GP for a in seq) / L
    return np.array([float(L), hyd, arom, pos, neg, net, gp], dtype=np.float64)


def build_X_for_bundle(
    seqs: List[str],
    bundle: dict,
    kmer_vectorizer,
) -> sp.csr_matrix:
    """
    Build X = [X_kmer | X_phys_scaled] to match training.
    Requires a fitted CountVectorizer passed in from CLI.
    """
    n_p = int(bundle["n_phys_features"])
    if n_p != 7:
        raise ValueError(
            f"Bundle expects n_phys_features={n_p}, but this script implements 7.\n"
            f"Your training phys features must match phys_features_7()."
        )
    X_kmer = kmer_vectorizer.transform(seqs).tocsr()
    X_phys = np.vstack([phys_features_7(s) for s in seqs])
    scaler = bundle["phys_scaler"]
    X_phys_scaled = scaler.transform(X_phys)
    X_phys_sp = sp.csr_matrix(X_phys_scaled)
    return sp.hstack([X_kmer, X_phys_sp], format="csr")


def proba_positive(bundle: dict, X: sp.csr_matrix) -> np.ndarray:
    return bundle["model"].predict_proba(X)[:, 1].astype(np.float64)


def percentile_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
    return ranks


@dataclass
class EnsembleConfig:
    w_neutral: float = 1.0
    w_select: float = 1.0
    w_bbb: float = 1.0
    gate_select_min: Optional[float] = 0.60
    gate_bbb_min: Optional[float] = 0.50
    rank_by: str = "combined"


class EnsembleRanker:
    def __init__(
        self,
        neutral_bundle: dict,
        select_bundle: dict,
        bbb_model: BBBProxyModel,
        kmer_vectorizer,
        cfg: EnsembleConfig,
    ):
        self.nb = neutral_bundle
        self.sb = select_bundle
        self.bbb = bbb_model
        self.vect = kmer_vectorizer
        self.cfg = cfg

    def score(self, seqs: List[str]) -> pd.DataFrame:
        seqs = [safe_seq(s) for s in seqs]
        Xn = build_X_for_bundle(seqs, self.nb, self.vect)
        Xs = build_X_for_bundle(seqs, self.sb, self.vect)
        neutral_raw = proba_positive(self.nb, Xn)
        select_raw = proba_positive(self.sb, Xs)
        bbb_raw = np.zeros(len(seqs), dtype=np.float64)
        feats_list: List[Dict[str, float]] = []
        for i, s in enumerate(seqs):
            if not s:
                bbb_raw[i] = -1e9
                feats_list.append({})
                continue
            score01, feats = self.bbb.score(s)
            bbb_raw[i] = float(score01)
            feats_list.append(feats)
        neutral_pct = percentile_rank(neutral_raw)
        select_pct = percentile_rank(select_raw)
        bbb_pct = percentile_rank(bbb_raw)
        combined = (
            self.cfg.w_neutral * neutral_pct
            + self.cfg.w_select * select_pct
            + self.cfg.w_bbb * bbb_pct
        ) / (self.cfg.w_neutral + self.cfg.w_select + self.cfg.w_bbb)

        def feat_col(key: str, default: float = float("nan")) -> np.ndarray:
            out = np.empty(len(feats_list), dtype=np.float64)
            for i, f in enumerate(feats_list):
                out[i] = float(f.get(key, default))
            return out

        df = pd.DataFrame(
            {
                "sequence": seqs,
                "neutralization_raw": neutral_raw,
                "selectivity_raw": select_raw,
                "bbb_raw": bbb_raw,
                "neutralization_pct": neutral_pct,
                "selectivity_pct": select_pct,
                "bbb_pct": bbb_pct,
                "combined": combined,
                "pI": feat_col("pI"),
                "charge_pH7p4": feat_col("charge_pH7p4"),
                "mw": feat_col("mw"),
                "gravy": feat_col("gravy"),
                "patch": feat_col("patch"),
                "bbb_raw_linear": feat_col("raw"),
            }
        )
        if self.cfg.gate_select_min is not None:
            df = df[df["selectivity_pct"] >= float(self.cfg.gate_select_min)]
        if self.cfg.gate_bbb_min is not None:
            df = df[df["bbb_pct"] >= float(self.cfg.gate_bbb_min)]
        if self.cfg.rank_by == "neutralization":
            df = df.sort_values(["neutralization_pct", "combined"], ascending=False)
        else:
            df = df.sort_values("combined", ascending=False)
        df = df.reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        return df


PROP_GROUPS_LIST = {
    "hydrophobic": list("AVILMFWY"),
    "polar": list("STNQ"),
    "positive": list("KRH"),
    "negative": list("DE"),
    "special": list("CGP"),
}

AA_GROUP: Dict[str, str] = {}
for g, letters in PROP_GROUPS_LIST.items():
    for a in letters:
        AA_GROUP[a] = g

GROUP_NEIGHBORS = {
    "hydrophobic": ["hydrophobic", "polar", "special"],
    "polar": ["polar", "hydrophobic", "positive", "negative", "special"],
    "positive": ["positive", "polar", "hydrophobic"],
    "negative": ["negative", "polar", "hydrophobic"],
    "special": ["special", "polar", "hydrophobic"],
}

AA_BASE_W = {a: 1.0 for a in AA}
for a in "CW":
    AA_BASE_W[a] = 0.7
for a in "GP":
    AA_BASE_W[a] = 0.85


def choose_positions(seq: str, n: int) -> List[int]:
    L = len(seq)
    n = max(1, min(n, L))
    return random.sample(range(L), k=n)


def guided_substitution(orig: str, temperature: float = 1.0) -> str:
    if orig not in AA_SET:
        return random.choice(AA)
    g = AA_GROUP.get(orig, "polar")
    neigh = GROUP_NEIGHBORS.get(g, [g])
    group_w = []
    for idx, gg in enumerate(neigh):
        w = 1.0 / (1.0 + idx)
        w = w ** (1.0 / max(1e-6, temperature))
        group_w.append(w)
    group_w = np.array(group_w, dtype=np.float64)
    group_w /= group_w.sum()
    chosen_group = random.choices(neigh, weights=group_w.tolist(), k=1)[0]
    candidates = PROP_GROUPS_LIST[chosen_group].copy()
    if orig in candidates and len(candidates) > 1:
        candidates.remove(orig)
    w = np.array([AA_BASE_W[a] for a in candidates], dtype=np.float64)
    w /= w.sum()
    return random.choices(candidates, weights=w.tolist(), k=1)[0]


@dataclass
class MutationEvent:
    pos: int
    frm: str
    to: str
    group_from: str
    group_to: str


def mutate_with_trace(
    seq: str, min_mut: int, max_mut: int, temperature: float
) -> Tuple[str, List[MutationEvent]]:
    seq = safe_seq(seq)
    if not seq:
        return seq, []
    n_mut = random.randint(min_mut, max_mut)
    pos_list = choose_positions(seq, n_mut)
    s = list(seq)
    events: List[MutationEvent] = []
    for i in pos_list:
        orig = s[i]
        new = guided_substitution(orig, temperature=temperature)
        gf = AA_GROUP.get(orig, "unknown")
        gt = AA_GROUP.get(new, "unknown")
        events.append(
            MutationEvent(pos=i, frm=orig, to=new, group_from=gf, group_to=gt)
        )
        s[i] = new
    mut_seq = "".join(s)
    if mut_seq == seq:
        return mut_seq, []
    return mut_seq, events


def mutation_string(events: List[MutationEvent]) -> str:
    if not events:
        return ""
    parts = [f"{e.frm}{e.pos+1}{e.to}" for e in sorted(events, key=lambda x: x.pos)]
    return ";".join(parts)


def mutation_mask(parent: str, child: str) -> str:
    if len(parent) != len(child):
        return ""
    return "".join("^" if a != b else " " for a, b in zip(parent, child))


def evolve_with_lineage(
    seed_seqs: List[str],
    ranker: EnsembleRanker,
    generations: int,
    parents: int,
    elite: int,
    children_per_parent: int,
    topk_per_parent: int,
    min_mut: int,
    max_mut: int,
    temperature: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    baseline = ranker.score(seed_seqs)
    current = baseline["sequence"].head(parents).tolist()
    all_dfs = [baseline.assign(generation=0)]
    lineage_records: List[Dict[str, object]] = []
    for g in range(1, generations + 1):
        parent_df = ranker.score(current)
        elites = parent_df["sequence"].head(elite).tolist()
        candidate_set = set(elites)
        for p in current:
            kid_events: Dict[str, List[MutationEvent]] = {}
            kids: List[str] = []
            for _ in range(children_per_parent):
                child, events = mutate_with_trace(
                    p, min_mut=min_mut, max_mut=max_mut, temperature=temperature
                )
                if child and child != p:
                    kids.append(child)
                    kid_events[child] = events
            kids = list(set(kids))
            if not kids:
                continue
            scored_kids = ranker.score(kids)
            best = scored_kids["sequence"].head(topk_per_parent).tolist()
            candidate_set.update(best)
            for child in best:
                events = kid_events.get(child, [])
                lineage_records.append(
                    {
                        "generation": g,
                        "parent": p,
                        "child": child,
                        "n_mutations": len(events),
                        "mutations": mutation_string(events),
                        "mask": mutation_mask(p, child),
                        "parent_group": "",
                    }
                )
        cand = ranker.score(list(candidate_set)).assign(generation=g)
        all_dfs.append(cand)
        current = cand["sequence"].head(parents).tolist()
        temperature = max(0.6, temperature * 0.9)
    full = pd.concat(all_dfs, ignore_index=True)
    full = full.sort_values("combined", ascending=False).drop_duplicates(
        "sequence", keep="first"
    )
    full = full.sort_values("combined", ascending=False).reset_index(drop=True)
    full["rank_global"] = np.arange(1, len(full) + 1)
    lineage_df = pd.DataFrame(lineage_records)
    return full, lineage_df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--col", default="cdr3")
    ap.add_argument("--neutral_model", required=True)
    ap.add_argument("--select_model", required=True)
    ap.add_argument(
        "--kmer_vectorizer",
        required=True,
        help="Fitted CountVectorizer saved with joblib",
    )
    ap.add_argument("--out_baseline", default="baseline_ranked.csv")
    ap.add_argument("--out_mutants", default="mutants_ranked.csv")
    ap.add_argument("--out_lineage", default="mutation_lineage.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--w_neutral", type=float, default=1.0)
    ap.add_argument("--w_select", type=float, default=1.0)
    ap.add_argument("--w_bbb", type=float, default=1.0)
    ap.add_argument(
        "--gate_select_min",
        type=float,
        default=0.60,
        help="0..1 percentile; set -1 to disable",
    )
    ap.add_argument(
        "--gate_bbb_min",
        type=float,
        default=0.50,
        help="0..1 percentile; set -1 to disable",
    )
    ap.add_argument(
        "--rank_by", choices=["combined", "neutralization"], default="combined"
    )
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--parents", type=int, default=30)
    ap.add_argument("--elite", type=int, default=10)
    ap.add_argument("--children_per_parent", type=int, default=200)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_mut", type=int, default=1)
    ap.add_argument("--max_mut", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--no_evolve",
        action="store_true",
        help="Only score baseline; skip mutation evolution",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    df = pd.read_csv(args.input)
    if args.col not in df.columns:
        raise SystemExit(f"Column '{args.col}' not found. Columns: {list(df.columns)}")
    seqs = [safe_seq(s) for s in df[args.col].astype(str).tolist()]
    seqs = [s for s in seqs if len(s) >= 6]
    if not seqs:
        raise SystemExit("No valid sequences found after cleaning.")
    neutral_bundle = joblib.load(args.neutral_model)
    select_bundle = joblib.load(args.select_model)
    vect = joblib.load(args.kmer_vectorizer)
    bbb_model = BBBProxyModel(BBBProxyConfig())
    gate_select = None if args.gate_select_min < 0 else float(args.gate_select_min)
    gate_bbb = None if args.gate_bbb_min < 0 else float(args.gate_bbb_min)
    cfg = EnsembleConfig(
        w_neutral=args.w_neutral,
        w_select=args.w_select,
        w_bbb=args.w_bbb,
        gate_select_min=gate_select,
        gate_bbb_min=gate_bbb,
        rank_by=args.rank_by,
    )
    ranker = EnsembleRanker(neutral_bundle, select_bundle, bbb_model, vect, cfg)
    Path(args.out_baseline).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_mutants).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_lineage).parent.mkdir(parents=True, exist_ok=True)
    baseline = ranker.score(seqs)
    baseline.to_csv(args.out_baseline, index=False)
    print(f"[OK] wrote {args.out_baseline} (n={len(baseline)})")
    if args.no_evolve:
        return
    full, lineage_df = evolve_with_lineage(
        seed_seqs=baseline["sequence"].tolist(),
        ranker=ranker,
        generations=args.generations,
        parents=args.parents,
        elite=args.elite,
        children_per_parent=args.children_per_parent,
        topk_per_parent=args.topk,
        min_mut=args.min_mut,
        max_mut=args.max_mut,
        temperature=args.temperature,
    )
    full.to_csv(args.out_mutants, index=False)
    print(f"[OK] wrote {args.out_mutants} (unique evaluated n={len(full)})")
    lineage_df.to_csv(args.out_lineage, index=False)
    print(f"[OK] wrote {args.out_lineage} (edges n={len(lineage_df)})")
    print("\nTop 10 finalists (global):")
    cols = [
        "rank_global",
        "sequence",
        "combined",
        "neutralization_pct",
        "selectivity_pct",
        "bbb_pct",
        "neutralization_raw",
        "selectivity_raw",
        "bbb_raw",
        "pI",
        "charge_pH7p4",
        "mw",
        "gravy",
        "patch",
        "generation",
    ]
    cols = [c for c in cols if c in full.columns]
    print(full[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
