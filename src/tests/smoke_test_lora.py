"""LoRA smoke test: verifies wrap_with_lora + gradient routing.

Checks that gradients flow into LoRA adapters and heads, but that no
frozen base encoder parameter ends up with a nonzero gradient.

Run: python -m src.tests.smoke_test_lora
"""

from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data_prep.dataset import TriLevelDataset
from src.model.architecture import TriLevelFinancialModel
from src.model.losses import TriLevelLoss
from src.model.lora_wrap import wrap_with_lora

MASTER = Path(r"E:\Coding\Python\Capstone-Project30\data\silver_dataset_master_with_text.csv")
BACKBONE = "roberta-base"


def main():
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    ds = TriLevelDataset(MASTER, tok, max_length=512)
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    model = TriLevelFinancialModel(BACKBONE, num_macro=8, num_industry=11, num_entity_tags=11)
    model = wrap_with_lora(model, r=8, alpha=16, dropout=0.05)

    loss_fn = TriLevelLoss()
    batch = next(iter(dl))
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    losses = loss_fn(out, batch)
    print("Losses:")
    for k, v in losses.items():
        print(f"  {k:20s} {float(v.detach()):.4f}")
    losses["loss"].backward()
    print("backward OK")

    lora_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "lora_" in n and p.grad is not None
    )
    head_grad = sum(
        p.grad.abs().sum().item()
        for n, p in model.named_parameters()
        if "_head" in n and p.grad is not None
    )
    base_bad = sum(
        1
        for n, p in model.named_parameters()
        if "lora_" not in n and "_head" not in n
        and p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"LoRA adapter grad total: {lora_grad:.4f}")
    print(f"Head grad total:         {head_grad:.4f}")
    print(f"Frozen base params with nonzero grad: {base_bad}  (expect 0)")


if __name__ == "__main__":
    main()
