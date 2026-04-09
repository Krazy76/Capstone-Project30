"""Wrap the shared encoder with PEFT LoRA adapters.

Only model.encoder is wrapped; the three task heads remain fully trainable
(they are tiny and random-init, LoRA on them is strictly worse). Target
modules are auto-detected by scanning submodule names so the same function
handles RoBERTa, FinBERT, and any future backbone swap.
"""

from __future__ import annotations

import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Candidate target module names by backbone family.
_TARGET_CANDIDATES: list[tuple[str, ...]] = [
    ("query", "value"),      # BERT / RoBERTa / FinBERT
    ("q_proj", "v_proj"),    # Llama / Mistral / Gemma-family
    ("q_lin", "v_lin"),      # DistilBERT
]


def _detect_target_modules(encoder: nn.Module) -> tuple[str, ...]:
    present = {name.split(".")[-1] for name, _ in encoder.named_modules()}
    for candidate in _TARGET_CANDIDATES:
        if all(c in present for c in candidate):
            return candidate
    raise ValueError(
        f"Could not auto-detect LoRA target modules; "
        f"submodule leaves: {sorted(present)[:20]}..."
    )


def _count_params(module: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def wrap_with_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: tuple[str, ...] | None = None,
    verbose: bool = True,
) -> nn.Module:
    """In-place wrap model.encoder with LoRA adapters and return the model."""
    if not hasattr(model, "encoder"):
        raise AttributeError("wrap_with_lora expects model.encoder to exist.")

    if target_modules is None:
        target_modules = _detect_target_modules(model.encoder)
        if verbose:
            print(f"[lora_wrap] auto-detected target modules: {target_modules}")

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=list(target_modules),
    )

    if verbose:
        tr, tot = _count_params(model)
        print(f"[lora_wrap] before: trainable={tr:,} / total={tot:,}")

    model.encoder = get_peft_model(model.encoder, lora_cfg)

    if verbose:
        tr, tot = _count_params(model)
        pct = 100.0 * tr / tot
        print(f"[lora_wrap] after:  trainable={tr:,} / total={tot:,} ({pct:.3f}%)")
        head_tr = sum(
            p.numel()
            for n, p in model.named_parameters()
            if p.requires_grad and "_head" in n
        )
        print(f"[lora_wrap] head trainable params: {head_tr:,}")

    return model
