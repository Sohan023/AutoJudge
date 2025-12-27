import pandas as pd

df = pd.read_csv("data/raw/problems.csv")

# only text columns we care about
text_cols = [
    "title",
    "description",
    "input_description",
    "output_description"
]

df[text_cols] = df[text_cols].fillna("")

# combine all text into one column
df["full_text"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["input_description"] + " " +
    df["output_description"]
)

# keep only what we need
df = df[["full_text", "problem_class", "problem_score"]]

df.to_csv("data/processed/problems_clean.csv", index=False)

print("Created problems_clean.csv")
print(df["problem_class"].value_counts())
