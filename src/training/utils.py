"""
Run utilities shared across training and evaluation: seeding, device detection,
and checkpoint save/load.

Why this file exists: keeps train.py focused on the training loop itself;
these are mechanical concerns that are identical regardless of which model or
phase is training.
"""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + CUDA) RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return CUDA if available, else CPU — raises nothing, just degrades gracefully."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    """Save model/optimizer state plus the epoch and metrics that produced it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """
    Load a checkpoint in place into model (and optimizer, if provided).

    Returns the raw checkpoint dict so callers can read epoch/metrics without
    a second disk read.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why seed python/numpy/torch separately instead of relying on one global seed?
# A1: They're independent RNG implementations with no shared state — PyTorch's
#     DataLoader shuffling, numpy-based augmentations, and any plain-python
#     random calls (e.g. in a custom sampler) each need their own seed call or
#     they'll silently use unseeded randomness even though "the seed" was set.
#
# Q2: Why save the optimizer state in checkpoints, not just model weights?
# A2: Adam/AdamW maintain per-parameter momentum and variance estimates; resuming
#     training with a fresh optimizer (zeroed moments) after a crash causes a
#     visible loss spike as those statistics re-accumulate. Saving optimizer
#     state makes resume-from-checkpoint behave like training was never
#     interrupted.
#
# Q3: Why device-agnostic (`cuda` if available else `cpu`) rather than hard
#     failing if no GPU is found?
# A3: ResNet-50 training is impractical on CPU, but evaluate.py and later the
#     FastAPI serving code (Phase 5) reuse this same get_device() — graceful
#     CPU fallback matters there (e.g. local smoke-testing the API without a
#     GPU), so this utility stays permissive and lets the caller decide what's
#     acceptable for its context.
