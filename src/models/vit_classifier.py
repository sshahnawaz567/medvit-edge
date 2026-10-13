"""
ViT-Base wrapper for multi-label chest X-ray classification — Phase 2.

Why this file exists: Phase 2's whole point is a fair full-fine-tune-vs-LoRA
comparison (PROJECT_BLUEPRINT.md Section 7, Phase 2). Both train_vit_full.py
and vit_lora.py/train_vit_lora.py wrap THIS SAME class — the only difference
between the two experiments is which parameters get gradients (all of them,
vs. a handful of LoRA adapters), not the architecture itself. If each used a
different wrapper, any AUROC/memory/speed difference we measured could just
be an architecture artifact instead of a genuine full-vs-LoRA effect.
"""

import torch
import torch.nn as nn
from transformers import ViTModel

# ImageNet-21k pretrained, not already fine-tuned on ImageNet-1k classes — we
# replace the head anyway, so starting from the broader 21k pretraining
# (rather than a checkpoint already specialized for 1k-way natural-image
# classification) gives the backbone's learned features a bit more generality
# to adapt from.
VIT_CHECKPOINT = "google/vit-base-patch16-224-in21k"


class ViTClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3, pretrained: bool = True):
        """
        Args:
            num_classes: 14 disease labels (same convention as BaselineResNet —
                "No Finding" is an all-zero vector, not a class).
            dropout: applied before the final linear layer. Higher than
                BaselineResNet's 0.2 (configs/baseline.yaml) — ViT-Base has
                ~86M params vs ResNet-50's ~25M, and Phase 1's own run
                (phases/phase1/LEARNINGS.md) already showed overfitting after
                epoch 4, so Phase 2 starts with stronger regularization
                up front rather than discovering the same curve again.
            pretrained: load ImageNet-21k pretrained weights. False gives a
                randomly-initialized ViT-Base — not used in Phase 2's actual
                experiments, but useful for fast architecture sanity-checks
                without a network call.
        """
        super().__init__()
        if pretrained:
            self.backbone = ViTModel.from_pretrained(VIT_CHECKPOINT)
        else:
            from transformers import ViTConfig
            self.backbone = ViTModel(ViTConfig())

        hidden_size = self.backbone.config.hidden_size  # 768 for ViT-Base
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: shape (batch, 3, 224, 224), already normalized —
                ViT has its own expected normalization stats, distinct from
                ResNet's ImageNet mean/std (see transforms.py's IMAGENET_MEAN/STD
                — Phase 2 will need its own transform for this reason).

        Returns:
            Raw logits, shape (batch, num_classes). Same convention as
            BaselineResNet: no sigmoid here, paired with BCEWithLogitsLoss
            in training.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        # ViT prepends a learned [CLS] token to the patch sequence specifically
        # so its final hidden state can summarize the whole image for
        # classification — same role nn.AdaptiveAvgPool2d + flatten plays at
        # the end of a CNN like ResNet, just learned instead of a fixed pool.
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


def freeze_backbone(model: ViTClassifier) -> None:
    """
    Freeze every backbone parameter in place (requires_grad=False), leaving
    only the classification head trainable.

    Why this matters for Phase 2: it's the cleanest way to isolate "how much
    does the backbone itself need to adapt" as a variable. It's also what
    distinguishes true LoRA fine-tuning from a frozen-backbone linear-probe
    baseline — LoRA adds a small number of trainable adapter parameters INSIDE
    the backbone's attention layers, so simply freezing everything is not the
    same as LoRA; this utility is for an ablation/comparison point, and for
    train_vit_full.py to optionally warm up with a few frozen-backbone epochs
    before unfreezing for full fine-tuning.

    Args:
        model: a ViTClassifier instance. Mutated in place; returns None.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why does ViT use a [CLS] token instead of global average pooling like a CNN?
# A1: ViT has no inherent spatial pooling hierarchy the way a CNN's stacked
#     conv/pool layers do — every patch token stays at the same resolution
#     through all transformer layers. The [CLS] token is prepended specifically
#     so self-attention can learn to aggregate whatever patches matter most
#     into one summary vector, rather than uniformly averaging all patches
#     (which would weight a diagnostically irrelevant corner of the X-ray the
#     same as the actual pathology).
#
# Q2: Why ImageNet-21k pretrained weights instead of the ImageNet-1k
#     fine-tuned checkpoint?
# A2: We discard whatever classification head came with the checkpoint either
#     way, so what matters is the backbone's learned features. The 21k
#     checkpoint was pretrained on a broader, more diverse label set before
#     any narrowing to 1k classes, which tends to transfer at least as well
#     to a new downstream domain (chest X-rays) as a backbone that was
#     additionally specialized for natural-image 1k-way classification.
#
# Q3: Why do ResNet-50 (Phase 1) and ViT-Base (Phase 2) need different image
#     normalization stats even though both are ImageNet-pretrained?
# A3: Each checkpoint was pretrained with whatever preprocessing pipeline its
#     original authors used — ResNet-50's torchvision weights expect the
#     classic ImageNet mean/std, while this particular ViT checkpoint has its
#     own expected normalization baked into its pretraining. Mismatching them
#     doesn't crash anything, it just silently feeds the backbone
#     out-of-distribution inputs relative to what it was pretrained on.
