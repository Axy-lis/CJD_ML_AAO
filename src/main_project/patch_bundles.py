import joblib

vect = joblib.load("models/kmer_vectorizer_3mer.joblib")

for p in ["models/neutralization_like.pkl", "models/selectivity_like.pkl"]:
    b = joblib.load(p)
    b["kmer_vectorizer"] = vect
    joblib.dump(b, p)
    print("Patched:", p)
