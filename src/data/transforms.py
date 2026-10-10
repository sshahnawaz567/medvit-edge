"""
Train/val/test image transforms for ResNet-50 on NIH ChestX-ray14.

Why this file exists: ResNet-50 was pretrained on ImageNet (3-channel RGB,
224x224, specific normalization statistics), but chest X-rays are natively
single-channel grayscale. This module centralizes the conversion + augmentation
policy so it's identical across train.py and evaluate.py.
"""

import torchvision.transforms as T

# ImageNet mean/std — required because we're using ImageNet-pretrained weights.
# If this project later trains a backbone from scratch, these should be replaced
# with statistics computed from the ChestX-ray14 training set itself.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int) -> T.Compose:
    """
    Training transforms: resize + mild augmentation + ImageNet normalization.

    Deliberately NOT using random horizontal flip: unlike natural images,
    chest X-rays are anatomically lateralized (e.g. the heart sits left of
    midline; dextrocardia — heart on the right — is itself a rare finding).
    Flipping would let the model "see" dextrocardia-like layouts as a normal
    augmentation of healthy patients, teaching the wrong prior. Published
    chest X-ray baselines (e.g. CheXNet) similarly avoid horizontal flipping.
    """
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.RandomRotation(degrees=10),        # small — patient positioning varies, pathology doesn't rotate
            T.ColorJitter(brightness=0.1, contrast=0.1),  # simulates exposure/detector variation across scanners
            T.Grayscale(num_output_channels=3),  # replicate the single channel to match ResNet's 3-channel input
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int) -> T.Compose:
    """Deterministic transforms for validation/test: no augmentation, same normalization as train."""
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why replicate grayscale to 3 channels instead of modifying ResNet's first
#     conv layer to accept 1 channel?
# A1: Either works, but replicating the channel lets us keep the pretrained
#     first-layer weights unmodified, so the ImageNet-pretrained low-level
#     edge/texture filters transfer directly. Modifying conv1 to 1 input
#     channel means either discarding those pretrained weights or averaging
#     them — simpler to replicate the image and keep the backbone untouched.
#
# Q2: Why is horizontal flip excluded when it's a near-default augmentation
#     for image classification?
# A2: Chest X-rays encode real anatomical laterality (heart position, aortic
#     arch side). Flipping can manufacture implausible or misleading anatomy
#     and risks teaching the model that left/right placement doesn't matter,
#     which is false for several of the 14 disease labels.
#
# Q3: Why ImageNet normalization stats instead of computing our own from
#     ChestX-ray14?
# A3: We're fine-tuning (not training from scratch) an ImageNet-pretrained
#     ResNet-50, so the backbone's early layers expect inputs distributed like
#     ImageNet's. Switching normalization stats without retraining those early
#     layers would shift the input distribution the pretrained filters were
#     calibrated for for no clear benefit at this stage.
