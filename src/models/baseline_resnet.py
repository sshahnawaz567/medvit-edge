"""
ResNet-50 baseline model for multi-label chest X-ray classification.

Why this file exists: Phase 1's goal is a reproducible baseline matching
published CheXNet-era performance (~0.82-0.84 mean AUROC) before any of the
ViT/LoRA/quantization work in later phases — this is the "known-good anchor"
every later optimization gets compared against.
"""

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class BaselineResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.2):
        """
        Args:
            num_classes: number of disease labels (14 for ChestX-ray14,
                "No Finding" is represented as an all-zero vector, not a class).
            pretrained: load ImageNet-pretrained backbone weights.
            dropout: dropout probability applied before the final linear layer,
                a light regularizer given ChestX-ray14's label noise (~10%,
                per PROJECT_BLUEPRINT.md Section 5.1).
        """
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        # Replace the 1000-way ImageNet classifier with a 14-way multi-label
        # head. We output raw logits (no sigmoid) because training pairs this
        # with BCEWithLogitsLoss, which applies sigmoid internally for
        # numerical stability (log-sum-exp trick) rather than us doing it
        # manually and risking float overflow in the log.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch, num_classes). Apply sigmoid for probabilities."""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why output raw logits instead of sigmoid probabilities from the model?
# A1: nn.BCEWithLogitsLoss combines sigmoid + binary cross-entropy in one
#     numerically stable operation (using the log-sum-exp trick), avoiding the
#     precision loss / potential NaNs from computing sigmoid then log(sigmoid(x))
#     separately when x is very negative. Sigmoid is applied only at inference
#     time, or inside the loss function during training.
#
# Q2: Why fine-tune the whole ResNet-50 end-to-end instead of freezing the
#     backbone and only training the new head?
# A2: Phase 1's explicit goal is a strong baseline (CheXNet-level ~0.82-0.84
#     mean AUROC), and chest X-ray features differ enough from natural
#     ImageNet images (different texture statistics, no color, different
#     spatial priors) that full fine-tuning typically outperforms a frozen
#     backbone for this domain shift. Freezing the backbone is explored later
#     in Phase 2 as part of the LoRA/full-fine-tune comparison.
#
# Q3: Why ResNet50_Weights.IMAGENET1K_V2 specifically, not V1 or random init?
# A3: V2 weights were trained with an improved torchvision recipe (better
#     augmentation, longer schedule) and consistently outperform V1 on
#     downstream transfer tasks at identical inference cost. Random init was
#     ruled out by Section 5.1's baseline target — matching CheXNet-era
#     results with limited local GPU time favors transfer learning over
#     training a 25M-parameter CNN from scratch.
