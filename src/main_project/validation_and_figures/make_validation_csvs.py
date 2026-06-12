#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN_PROJECT_DIR = HERE.parent

POSITIVES_CSV = MAIN_PROJECT_DIR / "positives.csv"
FDA_CSV = HERE / "fda.csv"
BACKGROUND_CSV = HERE / "background.csv"

CDR3_COL = "sequence"
NEUT_COL = "neutralization_raw"
SEL_COL = "selectivity_raw"

ID_COL = ""

BACKGROUND_SAMPLE_N = 20000
RANDOM_SEED = 42
OUTDIR = HERE / "results" / "validation_outputs"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_csv(path: str, label: str) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[ERROR] Cannot find {label} file at: {path}\n"
            "Check the repository layout."
        )
    return pd.read_csv(path)


def add_id(df: pd.DataFrame, group_prefix: str) -> pd.DataFrame:
    df = df.copy()
    if ID_COL and ID_COL in df.columns:
        df["id"] = df[ID_COL].astype(str)
    else:
        df["id"] = [f"{group_prefix}_{i:03d}" for i in range(1, len(df) + 1)]
    return df


def make_model_validation_df(
    pos: pd.DataFrame,
    fda: pd.DataFrame,
    bg: pd.DataFrame,
    score_col: str,
    out_csv_name: str,
    top_n: int = 50,
) -> pd.DataFrame:
    keep_cols = [CDR3_COL, score_col]
    for c in keep_cols:
        if c not in pos.columns:
            raise KeyError(
                f"[ERROR] Positives missing column '{c}'. Check CONFIG (CDR3_COL / score columns)."
            )
        if c not in fda.columns:
            raise KeyError(
                f"[ERROR] FDA missing column '{c}'. Check CONFIG (CDR3_COL / score columns)."
            )
        if c not in bg.columns:
            raise KeyError(
                f"[ERROR] Background missing column '{c}'. Check CONFIG (CDR3_COL / score columns)."
            )
    pos_s = pos[keep_cols].copy()
    fda_s = fda[keep_cols].copy()
    bg_s = bg[keep_cols].copy()
    pos_s["group"] = "Positive"
    fda_s["group"] = "FDA_NonPrion"
    bg_s["group"] = "Background"
    pos_s = add_id(pos_s, "POS")
    fda_s = add_id(fda_s, "FDA")
    bg_s = add_id(bg_s, "BG")
    pos_s = pos_s.rename(columns={score_col: "score"})
    fda_s = fda_s.rename(columns={score_col: "score"})
    bg_s = bg_s.rename(columns={score_col: "score"})
    df = pd.concat([pos_s, fda_s, bg_s], ignore_index=True)
    df = df.dropna(subset=[CDR3_COL, "score"]).reset_index(drop=True)
    df["rank_all"] = df["score"].rank(method="min", ascending=False).astype(int)
    bg_scores = df.loc[df["group"] == "Background", "score"].to_numpy()
    if len(bg_scores) == 0:
        df["percentile_vs_background"] = np.nan
    else:
        df["percentile_vs_background"] = df["score"].apply(
            lambda x: float(np.mean(bg_scores <= x))
        )
    df = df.sort_values(["score", "rank_all"], ascending=[False, True]).reset_index(
        drop=True
    )
    out_path = Path(OUTDIR) / out_csv_name
    df_out = df[
        ["id", CDR3_COL, "group", "score", "rank_all", "percentile_vs_background"]
    ].copy()
    df_out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} (n={len(df_out)})")
    top_path = Path(OUTDIR) / out_csv_name.replace(".csv", f"_TOP{top_n}.csv")
    df_out.head(top_n).to_csv(top_path, index=False)
    print(f"[OK] wrote {top_path} (top {top_n})")
    return df_out


def main():
    ensure_dir(OUTDIR)
    pos = load_csv(POSITIVES_CSV, "POSITIVES")
    fda = load_csv(FDA_CSV, "FDA")
    bg = load_csv(BACKGROUND_CSV, "BACKGROUND")
    if BACKGROUND_SAMPLE_N is not None and len(bg) > BACKGROUND_SAMPLE_N:
        bg = bg.sample(n=BACKGROUND_SAMPLE_N, random_state=RANDOM_SEED).reset_index(
            drop=True
        )
        print(f"[INFO] Sampled background to n={len(bg)} for speed")
    _ = make_model_validation_df(
        pos=pos,
        fda=fda,
        bg=bg,
        score_col=NEUT_COL,
        out_csv_name="validation_neutralization.csv",
        top_n=50,
    )
    _ = make_model_validation_df(
        pos=pos,
        fda=fda,
        bg=bg,
        score_col=SEL_COL,
        out_csv_name="validation_selectivity.csv",
        top_n=50,
    )
    print("\nDone. Use these in your report/slides:")
    print(f" - {OUTDIR / 'validation_neutralization.csv'}")
    print(f" - {OUTDIR / 'validation_selectivity.csv'}")


if __name__ == "__main__":
    main()
