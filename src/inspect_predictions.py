
import pandas as pd
from pathlib import Path

path = Path("results/baseline_predictions.csv")
df = pd.read_csv(path)

print("Total predictions:", len(df))
print("Missing reference reactants:", df["true_reactants"].isna().sum())
print("Missing predicted candidates:", df["predictions"].isna().sum())
print("Top-1 correct:", df["top1_correct"].sum())
print("\nFirst 5 examples:\n")

for _, row in df.head(5).iterrows():
    print("-" * 60)
    print("Patent ID:", row["id"])
    print("Product:", row["product_smiles"])
    print("True reactants:", row["true_reactants"])
    print("Predicted candidates:", row["predictions"])
    print("Top-1 correct:", row["top1_correct"])
