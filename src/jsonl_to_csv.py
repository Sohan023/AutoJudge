import json
import pandas as pd

rows = []

with open("data/raw/problems_data.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

# rename to match expected names
df = df.rename(columns={
    "difficulty": "problem_class",
    "score": "problem_score"
})

df.to_csv("data/raw/problems.csv", index=False)

print(df.columns.tolist())
print("Rows:", len(df))
