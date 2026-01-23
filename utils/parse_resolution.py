import pandas as pd

# CSV (авто-детект разделителя). Если точно TSV — замените sep=None на sep="\t".
df = pd.read_csv("datasets/data/annotations.csv", sep=None, engine="python")

# привести к числам (на случай строковых значений)
df["height"] = pd.to_numeric(df["height"], errors="coerce")
df["width"]  = pd.to_numeric(df["width"],  errors="coerce")

# уникальные пары
pairs = (df[["height","width"]]
         .dropna()
         .astype(int)
         .drop_duplicates()
         .sort_values(["height","width"]))

print(pairs.to_string(index=False))

# если нужны ещё и частоты каждой пары:
counts = (df.value_counts(["height","width"])
            .reset_index(name="count")
            .sort_values(["count","height","width"], ascending=[False, True, True]))
print("\nС частотами:\n", counts.to_string(index=False))
