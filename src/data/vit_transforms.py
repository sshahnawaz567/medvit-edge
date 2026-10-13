"""
Train/val/test image transforms for ViT-Base on NIH ChestX-ray14 — Phase 2.

Why this file exists (and isn't just reusing src/data/transforms.py): the
augmentation policy (resize, mild rotation, color jitter, no horizontal flip,
grayscale->3ch replication) is identical to Phase 1's reasoning and is
duplicated here deliberately. What CANNOT be reused is the normalization
constants — src/data/transforms.py's IMAGENET_MEAN/STD match what
torchvision's ResNet50_Weights.IMAGENET1K_V2 was pretrained with, but the
google/vit-base-patch16-224-in21k checkpoint (src/models/vit_classifier.py)
was pretrained with Hugging Face's ViTImageProcessor defaults instead:
mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] (i.e. scales pixels to [-1, 1],
not ImageNet statistics). Silently reusing Phase 1's transforms would still
run without error — it would just quietly feed the ViT backbone
out-of-distribution inputs relative to its pretraining, which is exactly the
kind of bug that doesn't show up until you're confused why AUROC is low.
Keeping this as its own file (rather than editing transforms.py) also
respects Phase 2's scope: Phase 1's files stay untouched.
"""

import torchvision.transforms as T

# google/vit-base-patch16-224-in21k's expected normalization (HF ViTImageProcessor
# default) — deliberately NOT src/data/transforms.py's IMAGENET_MEAN/STD. See
# module docstring above for why these differ.
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]


def get_vit_train_transforms(image_size: int) -> T.Compose:
    """
    Training transforms for ViT-Base: same augmentation policy as Phase 1
    (src/data/transforms.py get_train_transforms), different normalization.

    Horizontal flip is still deliberately excluded — chest X-ray laterality
    (heart position, aortic arch side) doesn't depend on which backbone
    architecture is consuming the image, so Phase 1's reasoning carries over
    unchanged (see transforms.py's Interview Notes Q2).
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ]
    )


def get_vit_eval_transforms(image_size: int) -> T.Compose:
    """Deterministic transforms for validation/test: no augmentation, ViT normalization."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=VIT_MEAN, std=VIT_STD),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Phase 1 and Phase 2 both fine-tune ImageNet-pretrained backbones on the
#     same images — why do they need different normalization constants?
# A1: "ImageNet-pretrained" only tells you the training data; it doesn't fix
#     the normalization convention, which each model's original authors chose
#     independently. Torchvision's ResNet-50 weights expect classic ImageNet
#     mean/std; Hugging Face's ViT checkpoint was preprocessed with a simple
#     [-1, 1] scaling (mean=std=0.5). Using the wrong one doesn't crash
#     anything — it just shifts every input out of the distribution the
#     backbone's early layers were calibrated for, silently hurting accuracy.
#
# Q2: Why duplicate the augmentation policy here instead of importing
#     get_train_transforms from transforms.py and only overriding the
#     Normalize step?
# A2: torchvision's T.Compose is a fixed list, not easily patched in place —
#     swapping one step out would mean reaching into transforms.py's internals
#     (`.transforms[-1] = ...`), which is more fragile and harder to read than
#     a second, explicit Compose. The duplication here is small (6 lines) and
#     keeps both files independently readable without import-time coupling
#     between phases.
#
# Q3: Why does ChestXrayDataset's grayscale-to-3-channel replication from
#     Phase 1 still apply here unchanged?
# A3: It's a property of the source data (NIH ChestX-ray14 is single-channel),
#     not of the backbone — ViT-Base, like ResNet-50, expects a 3-channel
#     input because it was pretrained on 3-channel natural images, so the same
#     fix (replicate the one real channel instead of discarding two of three
#     pretrained input-conv/patch-embedding channels) applies to both.
