# RetroSynthesisAI 🧪

## Transformer-Based Chemical Retrosynthesis Prediction Using SMILES and PyTorch

## Overview

RetroSynthesisAI is a machine learning project exploring the prediction of reactant molecules from a target product using SMILES representations. It implements a baseline prediction workflow and an experimental character-level Transformer using PyTorch.

Retrosynthesis is an important task in computational chemistry and drug discovery because it helps researchers explore possible starting materials for synthesizing a target molecule.

## Objectives

- Process and inspect reaction datasets.
- Establish a baseline for reactant prediction.
- Develop a character-level Transformer model using PyTorch.
- Evaluate the chemical validity and accuracy of generated reactant SMILES.
- Audit potential overlap between training, validation, and test datasets.
- Document model performance and limitations.

## Tech Stack

- **Python** — data processing and model development
- **PyTorch** — Transformer architecture and training
- **RDKit** — SMILES parsing and chemical validation
- **Pandas** — dataset processing
- **Scikit-learn** — baseline and evaluation utilities

## Project Structure

```text
RetroSynthesisAI/
├── src/
│   ├── audit_baseline.py
│   ├── baseline.py
│   ├── check_reactions.py
│   ├── data_overlap_audit.py
│   ├── evaluate.py
│   ├── inspect_data.py
│   ├── inspect_predictions.py
│   ├── preprocess.py
│   └── train_transformer.py
├── data/                 # Local datasets; not committed
├── checkpoints/          # Local model checkpoints; not committed
├── results/              # Local outputs
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

The processed dataset contains the following numbers of reactions:

| Split | Number of reactions |
|---|---:|
| Training | 40,008 |
| Validation | 5,001 |
| Test | 5,007 |

The dataset contains product SMILES and corresponding reactant SMILES. Dataset files are stored locally and are not committed to this repository.

An initial exact-string overlap audit identified 62 shared product strings between training and validation and 69 between training and test. These findings should be considered when interpreting evaluation results; canonicalized molecular overlap has not been established by this audit.

## Models

### 1. Baseline

A candidate-based baseline was evaluated on the 5,007-row test set. Its top-1 exact-string prediction matched 49 reference reactant strings.

### 2. Character-Level Transformer

An experimental sequence-to-sequence Transformer was implemented in PyTorch. It uses character-level SMILES tokens, positional encoding, teacher forcing during training, and greedy decoding during evaluation.

The initial local experiment used 1,000 training reactions and 200 validation reactions for five epochs. The best recorded validation loss was 1.4996. A separate evaluation then assessed the first 32 rows of the validation file.

## Experimental Results

### Baseline

| Metric | Result |
|---|---:|
| Test reactions evaluated | 5,007 |
| Top-1 exact-string matches | 49 |
| Top-1 exact-match accuracy | 0.98% |

### Transformer

| Metric | Result |
|---|---:|
| Validation reactions evaluated | 32 |
| Valid generated SMILES | 5 |
| Valid SMILES rate | 15.62% |
| Exact-string matches | 0 |
| Exact-match accuracy | 0.00% |

**Interpretation:** The initial Transformer experiment produced a low valid-SMILES rate and no exact-string matches in the small validation sample. These results do not demonstrate reliable retrosynthesis prediction.

The baseline and Transformer results were obtained on different evaluation samples and must not be interpreted as a direct performance comparison.

## Evaluation Metrics

- **Valid SMILES rate:** Fraction of generated predictions that RDKit can parse.
- **Exact-match accuracy:** Fraction of predictions matching the reference reactant string exactly.
- **Canonical reactant-set accuracy:** A planned metric that compares canonicalized molecular representations while accounting for multiple reactants.
- **Baseline comparison:** A comparison of prediction performance under the same evaluation conditions.

Canonical reactant-set accuracy has not yet been established for the reported Transformer experiment.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/dassreetama/RetroSynthesisAI.git
cd RetroSynthesisAI
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

RDKit and PyTorch must be installed successfully for the chemical validation and Transformer workflows.

### 4. Prepare the dataset

Obtain the dataset from its original provider, follow its access conditions and license, and place the processed files in the expected `data/processed/` directory.

### 5. Run the Transformer experiment

From the repository root:

```bash
python src/train_transformer.py
```

### 6. Evaluate the saved Transformer checkpoint

```bash
python src/evaluate.py
```

Check the scripts for their configured dataset paths, evaluation sample size, and checkpoint location before running them. The current evaluation script uses a small validation sample; it is not a full held-out test evaluation.

## Limitations

- The Transformer experiment uses a small training subset and has not established generalization to unseen reactions.
- Many generated SMILES strings are invalid.
- Exact-string matching can miss chemically equivalent representations.
- The baseline and Transformer results currently use different evaluation samples.
- Exact-string overlap does not detect every possible form of molecular overlap.
- Valid SMILES do not guarantee chemically plausible reactions.
- Predictions have not been experimentally verified.

## Future Work

- Train the Transformer on the full available training dataset.
- Improve SMILES tokenization, decoding, and chemical validity.
- Evaluate both models on the same held-out test set.
- Implement canonical reactant-set evaluation.
- Investigate dataset overlap using canonical molecular representations.
- Record reproducible configurations, metrics, and example predictions.

## Disclaimer

This project is intended for educational and research purposes. Generated predictions require independent chemical review before any practical use.