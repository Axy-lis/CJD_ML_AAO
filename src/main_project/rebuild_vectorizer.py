import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

t = pd.read_csv("features/targets_v1/rows.csv")
b = pd.read_csv("features/background_v1/rows_bg.csv")

seqs = (
    pd.concat([t["cdr3"], b["cdr3"]], ignore_index=True)
    .astype(str)
    .str.upper()
    .tolist()
)

vect = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False, min_df=1)
vect.fit(seqs)

joblib.dump(vect, "models/kmer_vectorizer_3mer.joblib")
print("Saved vocab size:", len(vect.vocabulary_))
