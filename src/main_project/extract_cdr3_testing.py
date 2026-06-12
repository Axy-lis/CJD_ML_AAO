from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

df = pd.read_csv(DATA_DIR / "airr_testing.csv")
cdr3 = df["cdr3_aa"].dropna().astype(str).str.upper()
cdr3 = cdr3[cdr3.str.len().between(10, 25)].drop_duplicates()

out_path = DATA_DIR / "testing_cdr3.csv"
pd.DataFrame({"cdr3": cdr3}).to_csv(out_path, index=False)

print(f"Saved {len(cdr3)} real CDR3 sequences to {out_path}")
