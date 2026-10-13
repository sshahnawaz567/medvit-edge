"""
Optimizer + scheduler builders for ViT-Base fine-tuning — Phase 2.

Why this file exists: Phase 1's train.py builds one AdamW instance with a
single learning rate for the whole model (reasonable for ResNet-50, where
backbone and head are both being fully trained from the same starting
point). Phase 2's full fine-tune (train_vit_full.py) needs a DIFFERENT
learning rate for the pretrained backbone vs the randomly-initialized
classifier head — see build_differential_lr_optimizer's docstring for why.
LoRA fine-tune (train_vit_lora.py) doesn't need this: peft's
apply_lora_to_vit (src/models/vit_lora.py) already restricts what's
trainable to the adapters + head, so a single learning rate across those
few trainable params is the standard LoRA setup and doesn't need splitting.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_differential_lr_optimizer(
    model: nn.Module,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Build an AdamW optimizer with two parameter groups: a low learning rate
    for the pretrained ViT backbone, a higher learning rate for the
    randomly-initialized classifier head. Full fine-tune only — LoRA fine-
    tune uses a single flat learning rate (see module docstring).

    Why differential LR matters here specifically: model.backbone starts
    from google/vit-base-patch16-224-in21k's pretrained weights, already
    reasonably well-calibrated; model.classifier starts from random init
    and needs to move much further, much faster, to become useful at all.
    A single LR high enough for the head to learn quickly risks destroying
    the backbone's pretrained features ("catastrophic forgetting") before
    the head has caught up; a single LR low enough to be safe for the
    backbone makes the head converge too slowly given Phase 2's tighter
    early_stopping_patience (configs/vit_full.yaml: 3, vs Phase 1's 5 —
    phases/phase1/LEARNINGS.md already documented ResNet-50 overfitting
    after epoch 4, so Phase 2 can't afford to spend early epochs with an
    undertrained head).

    Args:
        model: a ViTClassifier instance (src/models/vit_classifier.py) with
            model.backbone and model.classifier attributes.
        backbone_lr: learning rate for model.backbone's parameters
            (configs/vit_full.yaml, expected ~10x lower than head_lr).
        head_lr: learning rate for model.classifier's parameters.
        weight_decay: applied to both parameter groups identically — AdamW
            decouples weight decay from the LR itself, so this doesn't need
            the same 10x split the learning rates do.

    Returns:
        torch.optim.AdamW configured with two parameter groups (backbone,
        head), each at its own LR, both at the same weight_decay.

    """
    optimizer = AdamW(
        [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )

    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == backbone_lr
    assert optimizer.param_groups[1]["lr"] == head_lr

    return optimizer

def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Cosine-annealing scheduler shared by both train_vit_full.py and
    train_vit_lora.py — same choice and same reasoning as Phase 1's
    train.py (see its Interview Notes Q3), just factored out here since
    Phase 2 has two training scripts that both need it instead of one.

    Works unchanged whether optimizer has one parameter group (LoRA) or two
    (full fine-tune's differential LR) — CosineAnnealingLR decays every
    group's LR by the same multiplicative schedule, so the backbone/head LR
    ratio set in build_differential_lr_optimizer is preserved throughout
    training, not just at epoch 0.
    """
    return CosineAnnealingLR(optimizer, T_max=epochs)


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why does LoRA fine-tuning not need differential learning rates the way
#     full fine-tuning does?
# A1: Differential LR exists to protect already-well-calibrated pretrained
#     backbone weights from a head-sized learning rate. LoRA doesn't touch
#     those pretrained weights at all — it adds new, randomly-initialized
#     low-rank adapter matrices alongside them (src/models/vit_lora.py).
#     Those adapters start from scratch, same as the classifier head, so a
#     single LR across adapters + head is standard and there's no
#     pretrained-vs-random asymmetry left to protect against.
#
# Q2: Why does CosineAnnealingLR with two parameter groups preserve the
#     backbone/head LR ratio throughout training instead of letting them
#     drift apart?
# A2: CosineAnnealingLR computes a single decay multiplier per step from
#     T_max and the current step, then applies that SAME multiplier to every
#     parameter group's base LR. If backbone_lr starts at 10x lower than
#     head_lr, it stays at 10x lower (just both scaled down together) at
#     every point in the schedule — the ratio is a property of the two base
#     LRs, untouched by the scheduler.
#
# Q3: Why ~10x lower LR for the backbone rather than freezing it entirely
#     (freeze_backbone() in vit_classifier.py) during full fine-tune?
# A3: Freezing entirely is a different experiment (a linear-probe-style
#     ablation, useful for isolating "does the backbone need to move at
#     all") — full fine-tune's whole premise is letting the backbone adapt,
#     just cautiously. A small nonzero LR lets pretrained features shift
#     toward chest X-ray statistics without the large, disruptive updates a
#     head-sized LR would cause this early in training.
