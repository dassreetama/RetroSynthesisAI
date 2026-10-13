
from pathlib import Path
import time

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

DATA_DIR = Path("data/processed")
RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# Start with a subset so we can check the pipeline quickly.
# Change to None to evaluate the complete test set later.
TEST_LIMIT = None
TOP_K = 10

train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

# Remove rows with missing molecular strings.
train = train.dropna(
    subset=["product_smiles", "reactants_smiles"]
).reset_index(drop=True)

test = test.dropna(
    subset=["product_smiles", "reactants_smiles"]
).reset_index(drop=True)

if TEST_LIMIT is not None:
    test = test.head(TEST_LIMIT).copy()

# Use Morgan fingerprints to represent molecular structures.
generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048
)


def fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return generator.GetFingerprint(mol)


print("Generating training fingerprints...")
train_fps = []
valid_train = []

for _, row in train.iterrows():
    fp = fingerprint(row["product_smiles"])
    if fp is not None:
        train_fps.append(fp)
        valid_train.append(row)

train = pd.DataFrame(valid_train).reset_index(drop=True)

print("Usable training examples:", len(train))
print("Test examples:", len(test))

# Group equivalent reactant sets under the preprocessing convention.
def canonical_reactant_set(smiles):
    components = []

    for component in str(smiles).split("."):
        mol = Chem.MolFromSmiles(component)
        if mol is None:
            return None

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)

        components.append(
            Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=True
            )
        )

    return ".".join(sorted(components))


correct = {k: 0 for k in [1, 3, 5, 10]}
records = []
start = time.time()

for i, (_, row) in enumerate(test.iterrows(), start=1):
    query_fp = fingerprint(row["product_smiles"])

    if query_fp is None:
        continue

    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp, train_fps
    )

    # Highest similarity first; return up to ten candidates.
    ranked_indices = sorted(
        range(len(similarities)),
        key=lambda j: similarities[j],
        reverse=True
    )[:TOP_K]

    predictions = []
    seen = set()

    for j in ranked_indices:
        reactants = train.iloc[j]["reactants_smiles"]

        # Avoid duplicate candidates.
        key = canonical_reactant_set(reactants)
        if key is not None and key not in seen:
            seen.add(key)
            predictions.append({
                "reactants": key,
                "similarity": similarities[j]
            })

    truth = canonical_reactant_set(row["reactants_smiles"])
    ranked_sets = [p["reactants"] for p in predictions]

    if truth is not None:
        for k in correct:
            if truth in ranked_sets[:k]:
                correct[k] += 1

    records.append({
        "id": row["id"],
        "product_smiles": row["product_smiles"],
        "true_reactants": truth,
        "predictions": " || ".join(ranked_sets),
        "top1_similarity": (
            predictions[0]["similarity"] if predictions else None
        ),
        "top1_correct": bool(
            ranked_sets and truth == ranked_sets[0]
        ),
    })

    if i % 100 == 0:
        print(f"Processed {i}/{len(test)} test examples")

result = pd.DataFrame(records)
result.to_csv(RESULT_DIR / "baseline_predictions.csv", index=False)

n = len(result)

print("\nNearest-neighbor baseline results")
print("Evaluated examples:", n)

for k in correct:
    print(f"Top-{k} accuracy: {correct[k] / n:.4f}")

print("Prediction file: results/baseline_predictions.csv")
print(f"Elapsed time: {time.time() - start:.1f} seconds")
