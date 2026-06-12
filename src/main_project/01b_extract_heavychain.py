from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

df = pd.read_csv(DATA_DIR / "airr_raw.tsv", sep="\t")

bg = df[
    (df["locus"] == "IGH")
    & (df["productive"] == True)
    & (df["stop_codon"] == False)
    & (df["vj_in_frame"] == True)
].copy()

bg_full = bg[
    ["sequence_id", "sequence_aa", "cdr3_aa", "cdr3_start", "cdr3_end"]
].dropna(subset=["sequence_aa"])
bg_full.to_csv(DATA_DIR / "background_IGH_full.csv", index=False)

bg_cdr3 = (
    bg[["sequence_id", "cdr3_aa"]]
    .dropna(subset=["cdr3_aa"])
    .rename(columns={"cdr3_aa": "cdr3"})
)
bg_cdr3.to_csv(DATA_DIR / "background_IGH_cdr3.csv", index=False)
