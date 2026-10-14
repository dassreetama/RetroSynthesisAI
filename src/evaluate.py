
from pathlib import Path
import math

import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints" / "retro_transformer_test.pt"
TEST_FILE = ROOT / "data" / "processed" / "val.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PAD, BOS, EOS, UNK = 0, 1, 2, 3


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        positions = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(positions * divisor)
        pe[:, 1::2] = torch.cos(positions * divisor)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RetroTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.src_emb = nn.Embedding(
            config["vocab_size"], config["d_model"], padding_idx=PAD
        )
        self.tgt_emb = nn.Embedding(
            config["vocab_size"], config["d_model"], padding_idx=PAD
        )
        self.pos = PositionalEncoding(
            config["d_model"], config["max_len"]
        )
        self.transformer = nn.Transformer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_layers"],
            num_decoder_layers=config["num_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.out = nn.Linear(config["d_model"], config["vocab_size"])

    def forward(self, src, tgt):
        src_pad = src.eq(PAD)
        tgt_pad = tgt.eq(PAD)
        mask = torch.triu(
            torch.ones(
                tgt.size(1), tgt.size(1),
                dtype=torch.bool, device=tgt.device
            ),
            diagonal=1,
        )
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))
        hidden = self.transformer(
            src, tgt,
            tgt_mask=mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad,
        )
        return self.out(hidden)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT, map_location=DEVICE, weights_only=False
    )
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = checkpoint["config"]
    max_len = config["max_len"]

    model = RetroTransformer(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(TEST_FILE).dropna(
        subset=["product_smiles", "reactants_smiles"]
    ).head(32)

    valid_count = 0
    exact_count = 0

    print("Device:", DEVICE)
    print("Evaluation reactions:", len(df))
    print("\nSample predictions:\n")

    with torch.no_grad():
        for _, row in df.iterrows():
            product = str(row["product_smiles"])
            target = str(row["reactants_smiles"])

            src_ids = [BOS] + [
                stoi.get(ch, UNK) for ch in product
            ] + [EOS]
            src_ids = src_ids[:max_len]
            src = torch.tensor([src_ids], device=DEVICE)

            generated = [BOS]

            for _ in range(max_len - 1):
                tgt = torch.tensor([generated], device=DEVICE)
                logits = model(src, tgt)
                next_id = logits[0, -1].argmax().item()

                if next_id == EOS:
                    break
                if next_id == PAD:
                    break

                generated.append(next_id)

            prediction = "".join(
                itos[token]
                for token in generated[1:]
                if token >= 4
            )

            # RDKit checks whether the generated string parses as SMILES.
            molecule = Chem.MolFromSmiles(prediction)
            is_valid = molecule is not None
            valid_count += int(is_valid)

            exact_count += int(prediction == target)

            print("Product:   ", product)
            print("Predicted: ", prediction)
            print("Target:    ", target)
            print("Valid SMILES:", is_valid)
            print("-" * 60)

    total = len(df)
    print("\nEvaluation summary")
    print(f"Valid SMILES: {valid_count}/{total} ({100*valid_count/max(total,1):.2f}%)")
    print(f"Exact string matches: {exact_count}/{total} ({100*exact_count/max(total,1):.2f}%)")


if __name__ == "__main__":
    main()
