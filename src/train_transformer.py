
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------- Configuration ----------------

SEED = 42
MAX_LEN = 200
BATCH_SIZE = 16
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
FF_DIM = 512
DROPOUT = 0.1
EPOCHS = 5
LEARNING_RATE = 1e-4

# Small experiment: train on a subset, validate on a subset.
TRAIN_LIMIT = 1000
VAL_LIMIT = 200

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
CHECKPOINT_FILE = CHECKPOINT_DIR / "retro_transformer_test.pt"

PAD, BOS, EOS, UNK = 0, 1, 2, 3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ---------------- Load data ----------------

def load_data(path, limit=None):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    required = {"product_smiles", "reactants_smiles"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} must contain columns: {sorted(required)}"
        )

    df = df.dropna(
        subset=["product_smiles", "reactants_smiles"]
    ).copy()

    df["product_smiles"] = df["product_smiles"].astype(str)
    df["reactants_smiles"] = df["reactants_smiles"].astype(str)

    # Avoid truncating SMILES when adding BOS and EOS tokens.
    df = df[
        (df["product_smiles"].str.len() <= MAX_LEN - 2)
        & (df["reactants_smiles"].str.len() <= MAX_LEN - 2)
    ]

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


# Full training data is used to build the vocabulary.
full_train_df = load_data(TRAIN_FILE, limit=None)

# Only the first 1,000 reactions are used in this experiment.
train_df = full_train_df.head(TRAIN_LIMIT).copy()
val_df = load_data(VAL_FILE, limit=VAL_LIMIT)

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation dataset is empty.")


# ---------------- Build complete vocabulary ----------------

characters = set()

for column in ("product_smiles", "reactants_smiles"):
    for value in full_train_df[column]:
        characters.update(value)

itos = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"] + sorted(characters)
stoi = {token: index for index, token in enumerate(itos)}
VOCAB_SIZE = len(itos)

print("Full training dataset rows:", len(full_train_df))
print("Training rows used:", len(train_df))
print("Validation rows:", len(val_df))
print("Vocabulary size:", VOCAB_SIZE)
print("Vocabulary includes p:", "p" in stoi)
print("Vocabulary includes 7:", "7" in stoi)
print("Vocabulary includes 8:", "8" in stoi)

# Show any validation characters that are still unknown.
validation_characters = set()
for column in ("product_smiles", "reactants_smiles"):
    for value in val_df[column]:
        validation_characters.update(value)

missing_chars = sorted(validation_characters - set(characters))
print("Validation characters missing from training vocabulary:", missing_chars)


# ---------------- Encoding and batching ----------------

def encode(text):
    ids = [BOS]
    ids.extend(stoi.get(char, UNK) for char in text)
    ids.append(EOS)
    return ids


class ReactionDataset(Dataset):
    def __init__(self, frame):
        self.products = frame["product_smiles"].tolist()
        self.reactants = frame["reactants_smiles"].tolist()

    def __len__(self):
        return len(self.products)

    def __getitem__(self, index):
        return (
            encode(self.products[index]),
            encode(self.reactants[index]),
        )


def collate_batch(batch):
    sources, targets = zip(*batch)

    src_len = max(len(seq) for seq in sources)
    tgt_len = max(len(seq) for seq in targets)

    src = torch.full(
        (len(batch), src_len), PAD, dtype=torch.long
    )
    tgt = torch.full(
        (len(batch), tgt_len), PAD, dtype=torch.long
    )

    for i, seq in enumerate(sources):
        src[i, :len(seq)] = torch.tensor(seq)

    for i, seq in enumerate(targets):
        tgt[i, :len(seq)] = torch.tensor(seq)

    return src.to(DEVICE), tgt.to(DEVICE)


train_loader = DataLoader(
    ReactionDataset(train_df),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
)

val_loader = DataLoader(
    ReactionDataset(val_df),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
)


# ---------------- Transformer model ----------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).unsqueeze(1)

        divisor = torch.exp(
            torch.arange(0, d_model, 2)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(positions * divisor)
        pe[:, 1::2] = torch.cos(positions * divisor)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RetroTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.src_emb = nn.Embedding(
            VOCAB_SIZE, D_MODEL, padding_idx=PAD
        )
        self.tgt_emb = nn.Embedding(
            VOCAB_SIZE, D_MODEL, padding_idx=PAD
        )
        self.pos = PositionalEncoding(D_MODEL, MAX_LEN)

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True,
        )

        self.out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, src, tgt):
        src_padding = src.eq(PAD)
        tgt_padding = tgt.eq(PAD)

        causal_mask = torch.triu(
            torch.ones(
                tgt.size(1),
                tgt.size(1),
                dtype=torch.bool,
                device=tgt.device,
            ),
            diagonal=1,
        )

        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))

        hidden = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_padding,
            tgt_key_padding_mask=tgt_padding,
            memory_key_padding_mask=src_padding,
        )

        return self.out(hidden)


model = RetroTransformer().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)


# ---------------- Training and validation ----------------

def run_epoch(loader, training):
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    with torch.set_grad_enabled(training):
        for src, tgt in loader:
            decoder_input = tgt[:, :-1]
            labels = tgt[:, 1:]

            logits = model(src, decoder_input)

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1),
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            token_count = labels.ne(PAD).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    return total_loss / max(total_tokens, 1)


# ---------------- Save best checkpoint ----------------

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, training=True)
    val_loss = run_epoch(val_loader, training=False)

    print(
        f"Epoch {epoch}/{EPOCHS} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "stoi": stoi,
                "itos": itos,
                "config": {
                    "vocab_size": VOCAB_SIZE,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": FF_DIM,
                    "dropout": DROPOUT,
                    "max_len": MAX_LEN,
                },
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            CHECKPOINT_FILE,
        )

        print("Best checkpoint saved:", CHECKPOINT_FILE)

print("Experiment completed.")
print(f"Best validation loss: {best_val_loss:.4f}")
