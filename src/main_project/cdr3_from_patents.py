import os
import re
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
PATENT_DATA_DIR = SRC_DIR / "patent_data"
OUTPUT_DIR = SRC_DIR / "data" / "cdr3_from_patents"

FILES = [
    PATENT_DATA_DIR / "selectivity_patent2.csv",
    PATENT_DATA_DIR / "selectivity_patent1.csv",
    PATENT_DATA_DIR / "neutralization_patent1.csv",
    PATENT_DATA_DIR / "neutralization_patent2.csv",
    PATENT_DATA_DIR / "BBB_cross_patent1.csv",
]

CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


def read_table_smart(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    if header.count("\t") >= 3:
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def clean_aa(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^A-Z\*]", "", s)
    return s


def clean_dna(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().upper()
    s = re.sub(r"[^ACGT]", "", s)
    return s


def translate_best_frame(dna: str):
    dna = clean_dna(dna)
    if len(dna) < 60:
        return "", None
    best_aa, best_frame, best_score = "", None, -1e9
    for frame in (0, 1, 2):
        aa = []
        for i in range(frame, len(dna) - 2, 3):
            aa.append(CODON_TABLE.get(dna[i : i + 3], "X"))
        aa = "".join(aa)
        aa = aa.split("*")[0]
        if len(aa) < 60:
            continue
        score = len(aa) - 2 * aa.count("X")
        if score > best_score:
            best_score = score
            best_aa = aa
            best_frame = frame + 1
    return best_aa, best_frame


def extract_cdr3(seq_aa: str, chain_hint: str = ""):
    """
    Heuristic extractor:
      - find candidate C positions near the end
      - try end motifs in priority order:
          heavy: W[GQ]xG / WxxG / W
          light: FG (kappa/lambda J) / WxxG / W
      - pick the best candidate by a small scoring rule
    """
    seq = clean_aa(seq_aa)
    if not seq:
        return "", "empty"
    seq = seq.split("*")[0]
    chain = str(chain_hint).lower() if chain_hint is not None else ""
    is_heavy = any(k in chain for k in ["heavy", "igh", "vh"])
    if 4 <= len(seq) <= 30 and seq.isalpha():
        return seq, "as_is_short"
    tail_start = max(0, len(seq) - 220)
    tail = seq[tail_start:]
    c_positions = [tail_start + m.start() for m in re.finditer("C", tail)]
    if not c_positions:
        return "", "no_C"
    candidates = []
    for c_pos in c_positions:
        after = seq[c_pos + 1 :]
        if is_heavy:
            end_patterns = [
                (r"W[GQ][A-Z]G", "W_GQxG"),
                (r"W[A-Z]{0,2}G", "W_xxG"),
                (r"W", "W"),
            ]
            target_len = 15
        else:
            end_patterns = [
                (r"FG", "FG"),
                (r"W[A-Z]{0,2}G", "W_xxG"),
                (r"W", "W"),
            ]
            target_len = 13
        for pat, tag in end_patterns:
            m = re.search(pat, after)
            if not m:
                continue
            end_idx = c_pos + 1 + m.start()
            cdr3 = seq[c_pos + 1 : end_idx]
            if not (4 <= len(cdr3) <= 45):
                continue
            near_end = c_pos / max(1, len(seq))
            length_score = -abs(len(cdr3) - target_len) / target_len
            motif_bonus = 0.25 if tag == "FG" else 0.0
            score = near_end + length_score + motif_bonus
            candidates.append((score, cdr3, f"C_to_{tag}"))
    if not candidates:
        return "", "no_valid_span"
    best = max(candidates, key=lambda x: x[0])
    return best[1], best[2]


def pick_seq_column(df: pd.DataFrame):
    for col in [
        "sequence_aa",
        "sequence_aa ",
        "sequence",
        "sequence_aa_full",
        "sequence_aa_var",
    ]:
        if col in df.columns:
            return col
    return None


def pick_chain_column(df: pd.DataFrame):
    for col in ["chain", "chain_type", "chain_guess"]:
        if col in df.columns:
            return col
    return None


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
summary_rows = []

for path in FILES:
    name = path.stem
    df = read_table_smart(path)
    seq_col = pick_seq_column(df)
    chain_col = pick_chain_column(df)
    if "region" in df.columns and "sequence_aa" in df.columns:

        def cdr3_from_row(r):
            if str(r.get("region", "")).upper() == "CDR3":
                return clean_aa(r.get("sequence_aa", "")), "region==CDR3"
            return extract_cdr3(
                r.get("sequence_aa", ""), r.get(chain_col, "") if chain_col else ""
            )

        tmp = df.apply(lambda r: cdr3_from_row(r), axis=1, result_type="expand")
        df["cdr3"] = tmp[0]
        df["cdr3_method"] = tmp[1]
    elif "cdr3" in df.columns:
        df["cdr3"] = (
            df["cdr3"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.replace(r"[^A-Z]", "", regex=True)
        )
        df["cdr3_method"] = df.get("cdr3_method", "precomputed")
    else:
        if "sequence_type" in df.columns and "sequence" in df.columns:
            aa_list, frame_list = [], []
            for s in df["sequence"].tolist():
                aa, frame = translate_best_frame(s)
                aa_list.append(aa)
                frame_list.append(frame)
            df["translated_aa"] = aa_list
            df["translation_frame"] = frame_list
            tmp = df.apply(
                lambda r: extract_cdr3(
                    r["translated_aa"], r.get(chain_col, "") if chain_col else ""
                ),
                axis=1,
                result_type="expand",
            )
            df["cdr3"] = tmp[0]
            df["cdr3_method"] = "DNA_translate_then_" + tmp[1].astype(str)
        else:
            if not seq_col:
                raise ValueError(f"{path}: couldn't find a sequence column to use.")
            tmp = df.apply(
                lambda r: extract_cdr3(
                    r.get(seq_col, ""), r.get(chain_col, "") if chain_col else ""
                ),
                axis=1,
                result_type="expand",
            )
            df["cdr3"] = tmp[0]
            df["cdr3_method"] = tmp[1]
    df["cdr3_len"] = df["cdr3"].fillna("").astype(str).str.len()
    df["cdr3_ok"] = df["cdr3_len"].between(4, 45) & (df["cdr3"] != "")
    out_with = OUTPUT_DIR / f"{name}_with_cdr3.csv"
    out_only = OUTPUT_DIR / f"{name}_cdr3_only.csv"
    df.to_csv(out_with, index=False)
    cdr3_only = df.loc[df["cdr3_ok"], ["cdr3"]].drop_duplicates().copy()
    cdr3_only.to_csv(out_only, index=False)
    summary_rows.append(
        {
            "file": path.name,
            "rows": len(df),
            "cdr3_ok_rows": int(df["cdr3_ok"].sum()),
            "unique_cdr3": int(cdr3_only.shape[0]),
            "saved_with_cdr3": str(out_with),
            "saved_cdr3_only": str(out_only),
        }
    )

summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))
