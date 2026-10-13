
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")
pred_path = Path("results/baseline_predictions.csv")

train = pd.read_csv(data_dir / "train.csv")
test = pd.read_csv(data_dir / "test.csv")
pred = pd.read_csv(pred_path)

# Check exact product overlap between train and test.
train_products = set(train["product_smiles"].dropna())
test_products = set(test["product_smiles"].dropna())
overlap = train_products.intersection(test_products)

print("=" * 60)
print("DATASET OVERLAP AUDIT")
print("Training rows:", len(train))
print("Test rows:", len(test))
print("Unique training products:", len(train_products))
print("Unique test products:", len(test_products))
print("Unique products appearing in both:", len(overlap))
print("=" * 60)

# Inspect actual predictions.
print("\nFIRST 5 PREDICTIONS")

for _, row in pred.head(5).iterrows():
    print("\n" + "-" * 50)
    print("Patent ID:", row["id"])
    print("Product:", row["product_smiles"])
    print("True reactants:", row["true_reactants"])
    print("Predicted candidates:", row["predictions"])
    print("Top-1 correct:", row["top1_correct"])

# Summarize prediction availability.
print("\nPREDICTION SUMMARY")
print("Prediction rows:", len(pred))
print("Rows with no candidates:", pred["predictions"].isna().sum())
print("Top-1 correct:", pred["top1_correct"].sum())
