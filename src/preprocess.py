
from pathlib import Path
import pandas as pd
from rdkit import Chem

DATA_DIR = Path("data")
OUTPUT_DIR = DATA_DIR / "processed"

REACTION_COLUMN = "reactants>reagents>production"
SPLITS = ["train", "val", "test"]


def canonicalize_smiles(smiles):
    """Canonicalize a molecule and remove atom-map numbers."""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )


def process_reaction(reaction):
    """Convert reactants>>product into product/reactant fields."""
    parts = str(reaction).split(">>")

    if len(parts) != 2:
        return None

    reactant_text, product_text = parts

    product = canonicalize_smiles(product_text)

    if not product:
        return None

    reactants = []

    for component in reactant_text.split("."):
        component = component.strip()

        if not component:
            continue

        canonical = canonicalize_smiles(component)

        if canonical is None:
            return None

        reactants.append(canonical)

    if not reactants:
        return None

    # Consistent ordering of multi-component reactant sets
    reactants.sort()

    return {
        "product_smiles": product,
        "reactants_smiles": ".".join(reactants),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        input_path = DATA_DIR / f"raw_{split}.csv"
        output_path = OUTPUT_DIR / f"{split}.csv"

        df = pd.read_csv(input_path)

        required = {
            "id",
            "class",
            REACTION_COLUMN,
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{input_path} is missing columns: {missing}"
            )

        records = []
        rejected = 0

        for _, row in df.iterrows():
            result = process_reaction(
                row[REACTION_COLUMN]
            )

            if result is None:
                rejected += 1
                continue

            records.append({
                "id": row["id"],
                "class": row["class"],
                **result,
            })

        processed = pd.DataFrame(
            records,
            columns=[
                "id",
                "class",
                "product_smiles",
                "reactants_smiles",
            ],
        )

        processed.to_csv(output_path, index=False)

        print(f"\nSplit: {split}")
        print(f"Raw rows: {len(df)}")
        print(f"Processed rows: {len(processed)}")
        print(f"Rejected rows: {rejected}")
        print(f"Saved to: {output_path}")

        if not processed.empty:
            print(processed.head(2).to_string(index=False))

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
