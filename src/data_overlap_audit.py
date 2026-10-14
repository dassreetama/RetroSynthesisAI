
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "val.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

required = {"product_smiles", "reactants_smiles"}
for name, df in [("train", train), ("val", val), ("test", test)]:
    if not required.issubset(df.columns):
        raise ValueError(f"{name} is missing required columns: {required - set(df.columns)}")

def product_set(df):
    return set(df["product_smiles"].dropna().astype(str).str.strip())

train_products = product_set(train)
val_products = product_set(val)
test_products = product_set(test)

print("DATASET OVERLAP AUDIT")
print("-" * 45)
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(val)}")
print(f"Test rows: {len(test)}")
print(f"Unique training products: {len(train_products)}")
print(f"Unique validation products: {len(val_products)}")
print(f"Unique test products: {len(test_products)}")

for name, products in [("validation", val_products), ("test", test_products)]:
    overlap = train_products & products
    print(f"\nTrain-{name} overlapping products: {len(overlap)}")
    print(f"Train-{name} overlap percentage: "
          f"{100 * len(overlap) / max(len(products), 1):.2f}%")

print("\nNote: exact string overlap only; chemical canonicalization is not performed.")
