
from pathlib import Path
import pandas as pd
from rdkit import Chem

DATA_DIR = Path("data")
REACTION_COLUMN = "reactants>reagents>production"

for split in ["train", "val", "test"]:
    path = DATA_DIR / f"raw_{split}.csv"
    df = pd.read_csv(path)

    print(f"\n{'=' * 55}")
    print(f"Split: {split}")
    print(f"Rows: {len(df)}")
    print("Reaction classes:")
    print(df["class"].value_counts(dropna=False).head(15))

    valid_reactions = 0
    valid_products = 0
    invalid_products = 0
    malformed = 0

    for reaction in df[REACTION_COLUMN].dropna():
        # Standard reaction SMILES: reactants>>product
        parts = str(reaction).split(">>")

        if len(parts) != 2:
            malformed += 1
            continue

        reactants, product = parts

        product_mol = Chem.MolFromSmiles(product)
        if product_mol is not None:
            valid_products += 1
        else:
            invalid_products += 1

        reactant_mols = [
            Chem.MolFromSmiles(s)
            for s in reactants.split(".")
            if s.strip()
        ]

        if (
            product_mol is not None
            and reactant_mols
            and all(mol is not None for mol in reactant_mols)
        ):
            valid_reactions += 1

    print("Valid products:", valid_products)
    print("Invalid products:", invalid_products)
    print("Fully valid reactions:", valid_reactions)
    print("Malformed reaction strings:", malformed)

print("\nValidation finished.")
