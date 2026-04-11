"""End-to-end smoke test: dataset -> model -> loss -> backward on one batch.

Run: python -m src.tests.smoke_test
"""

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_prep.dataset import TriLevelDataset
from src.model.architecture import TriLevelFinancialModel
from src.model.losses import TriLevelLoss

MASTER = Path(r"E:\Coding\Python\Capstone-Project30\data\silver_dataset_master_with_text.csv")
BACKBONE = "roberta-base"


def main():
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    ds = TriLevelDataset(MASTER, tok, max_length=512)
    print(f"dataset size: {len(ds)}")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    model = TriLevelFinancialModel(BACKBONE, num_macro=8, num_industry=11, num_entity_tags=11)
    loss_fn = TriLevelLoss()

    batch = next(iter(dl))
    print("Batch shapes:")
    for k, v in batch.items():
        print(f"  {k:20s} {tuple(v.shape)}  dtype={v.dtype}")

    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    print("Output shapes:")
    for k, v in out.items():
        print(f"  {k:20s} {tuple(v.shape)}")

    losses = loss_fn(out, batch)
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k:20s} {float(v.detach()):.4f}")

    losses["loss"].backward()
    print("backward OK")

    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )
    print(f"encoder has gradient: {has_grad}")


if __name__ == "__main__":
    main()
