import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
import joblib

print("Loading features and labels...")

X = pd.read_csv("data/processed/features.csv").values
y = pd.read_csv("data/processed/labels.csv")

texts = pd.read_csv(
    "data/processed/problems_clean.csv"
)["full_text"].astype(str).values

y_class = y["problem_class"].map({
    "easy": 0,
    "medium": 1,
    "hard": 2
})

y_score = y["problem_score"]

X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test, texts_train, texts_test = train_test_split(
    X,
    y_class,
    y_score,
    texts,
    test_size=0.2,
    random_state=42,
    stratify=y_class
)

print("Training classifier...")

clf = RandomForestClassifier(
    n_estimators=800,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_class_train)

print("Training regressor...")

reg = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

reg.fit(X_train, y_score_train)

print("Evaluating models...")

y_pred = clf.predict(X_test)
y_pred_final = y_pred.copy()

# ================== RULE-BASED STRATIFICATION ==================

HARD_KEYWORDS = [
    "graph", "tree", "dfs", "bfs",
    "shortest path", "dijkstra",
    "segment tree", "fenwick",
    "union find", "disjoint set",
    "bitmask", "bit manipulation",
    "dynamic programming", "dp",
    "topological", "bellman",
    "binary lifting", "scc"
]

EASY_KEYWORDS = [
    "print", "output",
    "simple", "basic",
    "brute force",
    "loop", "iteration",
    "sum", "count",
    "array", "string",
    "math"
]

for i, text in enumerate(texts_test):
    t = text.lower()

    hard_score = sum(k in t for k in HARD_KEYWORDS)
    easy_score = sum(k in t for k in EASY_KEYWORDS)
    length = len(t.split())

    if (
        hard_score >= 3 or
        (hard_score >= 2 and length > 400) or
        ("constraints" in t and hard_score >= 2)
    ):
        y_pred_final[i] = 2
        continue

    if (
        easy_score >= 3 and
        hard_score == 0 and
        length < 200
    ):
        y_pred_final[i] = 0
        continue

# ===============================================================

acc = accuracy_score(y_class_test, y_pred_final)
mae = mean_absolute_error(y_score_test, reg.predict(X_test))
rmse = np.sqrt(mean_squared_error(y_score_test, reg.predict(X_test)))

print("\nClassification Accuracy:", acc)
print(
    classification_report(
        y_class_test,
        y_pred_final,
        target_names=["easy", "medium", "hard"]
    )
)

print("Regression MAE:", mae)
print("Regression RMSE:", rmse)

joblib.dump(clf, "data/processed/classifier.pkl")
joblib.dump(reg, "data/processed/regressor.pkl")

print("Models saved: classifier.pkl, regressor.pkl")
print("DONE")
