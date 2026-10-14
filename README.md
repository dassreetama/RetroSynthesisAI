# RetroSynthesisAI 🧪

**Transformer-Based Chemical Retrosynthesis Prediction Using SMILES and PyTorch**

## Overview

RetroSynthesisAI is a machine learning project exploring the prediction of reactant molecules from a target product using SMILES representations and a Transformer-based sequence-to-sequence model.

Retrosynthesis is an important task in computational chemistry and drug discovery because it helps researchers explore possible starting materials for synthesizing a target molecule.

## Objectives

- Process and inspect reaction datasets.
- Establish a baseline for reactant prediction.
- Develop a character-level Transformer model using PyTorch.
- Evaluate the chemical validity and accuracy of generated reactant SMILES.
- Compare Transformer predictions against a baseline.

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
│   ├── evaluate.py
│   ├── inspect_data.py
│   ├── inspect_predictions.py
│   ├── preprocess.py
│   └── train_transformer.py
├── data/               # Local datasets; not committed
├── results/            # Local outputs
├── .gitignore
├── README.md
└── requirements.txt
```

## Current Status

**Work in progress.**

The repository includes data inspection and preprocessing scripts, a baseline workflow, and initial experiments with a character-level Transformer on a small subset of reactions.

Full-dataset training and held-out test evaluation are still pending. Final predictive performance has not yet been established.

## Evaluation Plan

The model will be evaluated using:

- **Valid SMILES rate:** fraction of predictions that RDKit can parse.
- **Exact-match accuracy:** fraction of predictions matching the reference reactant string.
- **Canonical reactant-set accuracy:** comparison after canonicalization and appropriate handling of multiple reactants.
- **Baseline comparison:** performance relative to a simpler prediction method.

## Getting Started

Create and activate a Python virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The dataset must be obtained separately from its original provider. Follow its access conditions and license.

Run scripts from the repository root using the commands appropriate to their implemented arguments. Training and evaluation instructions will be updated as the workflow is finalized.

## Limitations

- Predicted reactions are not experimentally verified.
- Valid SMILES do not guarantee chemically plausible reactions.
- Performance on a small training subset does not establish generalization to unseen reactions.

## Future Work

- Complete full-dataset Transformer training.
- Evaluate on held-out test reactions.
- Compare against the baseline.
- Document reproducible results and example predictions.

## Disclaimer

This project is intended for educational and research purposes. Predictions require independent chemical review before practical use.
