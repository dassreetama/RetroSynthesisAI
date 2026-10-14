
from pathlib import Path
import pandas as pd
from rdkit import Chem

DATA_DIR = Path("data/processed")

def check_split(split):
    path = DATA_DIR / f"{split}.csv"

    if not path.exists():
        print(f"\n{split.upper()}: FILE NOT FOUND — {path}")
        return

    df = pd.read_csv(path)
    print(f"\n{'=' * 50}")
    print(f"Split: {split}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    required = {"product_smiles", "reactants_smiles"}
    if not required.issubset(df.columns):
        print("ERROR: Required SMILES columns are missing.")
        return

    missing_products = df["product_smiles"].isna().sum()
    missing_reactants = df["reactants_smiles"].isna().sum()

    print(f"Missing products: {missing_products}")
    print(f"Missing reactants: {missing_reactants}")

    valid_products = 0
    valid_reactants = 0
    valid_reactions = 0

    for product, reactants in zip(
        df["product_smiles"], df["reactants_smiles"]
    ):
        product_ok = (
            isinstance(product, str)
            and Chem.MolFromSmiles(product) is not None
        )

        # Each dot-separated component must parse.
        reactant_ok = (
            isinstance(reactants, str)
            and bool(reactants.strip())
            and all(
                Chem.MolFromSmiles(part) is not None
                for part in reactants.split(".")
            )
        )

        valid_products += int(product_ok)
        valid_reactants += int(reactant_ok)
        valid_reactions += int(product_ok and reactant_ok)

    print(f"Valid products: {valid_products}/{len(df)}")
    print(f"Valid reactant strings: {valid_reactants}/{len(df)}")
    print(f"Fully valid reactions: {valid_reactions}/{len(df)}")

for split in ["train", "val", "test"]:
    check_split(split)

print("\nDataset validation finished.")
