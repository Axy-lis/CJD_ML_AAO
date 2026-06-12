from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).resolve().parent

POS_NEUT_CSV = HERE / "positives_trimmed_neut.csv"
POS_SPEC_CSV = HERE / "positives_trimmed_select.csv"
FDA_CSV = HERE / "fda.csv"
BACKGROUND_CSV = HERE / "background.csv"

NEUT_COL = "neutralization_raw"
SPEC_COL = "selectivity_raw"


def main() -> None:
    pos_neut = pd.read_csv(POS_NEUT_CSV)
    pos_spec = pd.read_csv(POS_SPEC_CSV)
    fda = pd.read_csv(FDA_CSV)
    bg = pd.read_csv(BACKGROUND_CSV)
    if len(bg) > 20000:
        bg = bg.sample(n=20000, random_state=42)
    positive_neutralization_scores = pos_neut[NEUT_COL].dropna().values
    background_neutralization_scores = bg[NEUT_COL].dropna().values
    fda_neutralization_scores = fda[NEUT_COL].dropna().values
    positive_specificity_scores = pos_spec[SPEC_COL].dropna().values
    background_specificity_scores = bg[SPEC_COL].dropna().values
    fda_specificity_scores = fda[SPEC_COL].dropna().values
    print("Neutralization stats:")
    print("Pos mean:", np.mean(positive_neutralization_scores))
    print("BG mean:", np.mean(background_neutralization_scores))
    print("FDA mean:", np.mean(fda_neutralization_scores))
    print(
        "Pos min/max:",
        np.min(positive_neutralization_scores),
        np.max(positive_neutralization_scores),
    )
    print(
        "BG min/max:",
        np.min(background_neutralization_scores),
        np.max(background_neutralization_scores),
    )
    print("\n===== NEUTRALIZATION MODEL VALIDATION =====")
    u1, p1 = mannwhitneyu(
        positive_neutralization_scores,
        background_neutralization_scores,
        alternative="greater",
    )
    print(f"Positives (Neut) vs Background | U={u1:.1f} | p={p1:.4e}")
    u2, p2 = mannwhitneyu(
        positive_neutralization_scores,
        fda_neutralization_scores,
        alternative="greater",
    )
    print(f"Positives (Neut) vs FDA        | U={u2:.1f} | p={p2:.4e}")
    print("\n===== SPECIFICITY MODEL VALIDATION =====")
    u3, p3 = mannwhitneyu(
        positive_specificity_scores,
        background_specificity_scores,
        alternative="greater",
    )
    print(f"Positives (Spec) vs Background | U={u3:.1f} | p={p3:.4e}")
    u4, p4 = mannwhitneyu(
        positive_specificity_scores,
        fda_specificity_scores,
        alternative="greater",
    )
    print(f"Positives (Spec) vs FDA        | U={u4:.1f} | p={p4:.4e}")


if __name__ == "__main__":
    main()
