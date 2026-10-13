"""
LoRA configuration + application for ViT-Base — Phase 2's LoRA arm.

Why this file exists: Phase 2's core comparison is full fine-tune (all ~86M
ViT-Base params trainable, src/training/train_vit_full.py) vs LoRA (a small
set of low-rank adapter params trainable, src/training/train_vit_lora.py).
This file is the ONLY place that difference lives — both training scripts
otherwise share the exact same ViTClassifier (src/models/vit_classifier.py),
data pipeline, and metrics, so any AUROC/memory/speed delta measured between
the two runs can be attributed to "full vs LoRA," not to an architecture or
data pipeline difference.
"""

import torch.nn as nn
from peft import LoraConfig, get_peft_model

def apply_lora_to_vit(model, rank, alpha, target_modules, dropout=0.1):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why is apply_lora_to_vit() applied to a ViTClassifier instance rather
#     than to the raw Hugging Face ViTModel before wrapping it in a classifier?
# A1: The classification head (model.classifier) has no pretrained weights —
#     it's randomly initialized regardless of which fine-tuning strategy is
#     used, and it must stay fully trainable under both full fine-tune and
#     LoRA (LoRA only makes sense as a low-rank delta against an existing
#     pretrained weight matrix). Applying LoRA after the full ViTClassifier is
#     assembled keeps that head intact and only touches backbone attention
#     projections.
#
# Q2: What's actually being compared in "full fine-tune vs LoRA" if both
#     still train the classifier head fully?
# A2: Whether the backbone's pretrained features need updating at all to
#     adapt to chest X-rays, and if so, how much adaptation capacity is
#     actually required. Full fine-tune updates every backbone weight;
#     LoRA restricts backbone adaptation to a handful of low-rank matrices
#     in the attention projections. If LoRA gets within ~2% mean AUROC
#     (PROJECT_BLUEPRINT.md Section 7 success criterion) using a fraction of
#     the trainable parameters, that's evidence the domain shift
#     (natural images -> chest X-rays) doesn't require full-capacity
#     backbone adaptation.
#
# Q3: Why rank=8 specifically, and what would happen with rank=1 or rank=64?
# A3: Rank=8 is a common starting point in LoRA literature — enough capacity
#     for moderate domain shift without approaching full fine-tune's
#     parameter count. Rank=1 might underfit (not enough capacity to capture
#     the natural-image -> X-ray adaptation); rank=64 would narrow the
#     parameter-efficiency gap against full fine-tune while only being
#     justified if rank=8 demonstrably underperforms — this is exactly the
#     kind of hyperparameter a real ablation (not just one run) would sweep,
#     which is itself worth naming as a limitation in reports/phase2_comparison.md.
