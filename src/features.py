import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("Loading cleaned data...")

df = pd.read_csv("data/processed/problems_clean.csv")
texts = df["full_text"].astype(str)

print("Creating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=4000,
    stop_words="english",
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(texts).toarray()

ALGO_KEYWORDS = [
    "dp", "dynamic programming",
    "graph", "tree", "dfs", "bfs",
    "shortest path", "dijkstra", "bellman",
    "segment tree", "fenwick",
    "binary search",
    "two pointers", "sliding window",
    "subarray", "subsequence",
    "greedy",
    "backtracking", "recursion",
    "bitmask", "bit manipulation",
    "heap", "priority queue",
    "union find", "disjoint set",
    "topological",
    "math", "number theory",
    "modulo", "combinatorics"
]

def extra_features(t):
    t = t.lower()
    return [
        len(t),
        len(t.split()),
        t.count("\n"),
        sum(t.count(c) for c in "+-*/%=<>"),
        sum(1 for k in ALGO_KEYWORDS if k in t)
    ]

print("Extracting handcrafted features...")

extra = np.array([extra_features(t) for t in texts])

X = np.hstack([X_tfidf, extra])

tfidf_cols = [f"tfidf_{i}" for i in range(X_tfidf.shape[1])]
extra_cols = [
    "char_length",
    "word_count",
    "newline_count",
    "math_symbol_count",
    "algo_keyword_count"
]

cols = tfidf_cols + extra_cols

print("Saving features and labels...")

pd.DataFrame(X, columns=cols).to_csv(
    "data/processed/features.csv",
    index=False
)

df[["problem_class", "problem_score"]].to_csv(
    "data/processed/labels.csv",
    index=False
)

joblib.dump(tfidf, "data/processed/tfidf.pkl")

print("Features created:", X.shape)
print("features.csv, labels.csv, tfidf.pkl saved")
