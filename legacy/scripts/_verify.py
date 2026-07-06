import pandas as pd

df = pd.read_parquet("data/splits/test_augmented.parquet")
print("test_augmented.parquet satir sayisi:", len(df))
print("Kolonlar:", list(df.columns))
print()
print("Label dagilimi:")
print(df["label"].value_counts().to_string())
print()
print("Source dagilimi:")
print(df["source"].value_counts().head(10).to_string())
print()
mc = df[df["source"] == "manual_collection"]
print("manual_collection ornekleri:", len(mc))
print(mc["label"].value_counts().to_string())
print()
print("label_id kontrolu (gurur=9, utanc=8):")
print(mc[["label","label_id","label_norm"]].drop_duplicates().to_string())
