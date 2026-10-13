"""
Phase 2 training loop: ViT-Base FULL fine-tune on NIH ChestX-ray14.

Why this file exists: this is one half of Phase 2's full-vs-LoRA comparison
(train_vit_lora.py is the other half). The training mechanics — per-step
loss, per-epoch validation, checkpointing on best val mean AUROC, W&B
logging — are IDENTICAL to Phase 1's train.py, so this script imports and
reuses those functions directly instead of re-implementing them (the one
thing that must NOT differ between the two phases is "how training loops
work," so any AUROC delta we measure later reflects the model, not the
loop). What's genuinely new here: loading ViT instead of ResNet, a
differential-LR optimizer (see src/training/vit_optim.py — backbone needs a
much lower LR than the randomly-initialized head), ViT-specific
normalization (src/data/vit_transforms.py), and explicit trainable-param /
GPU-peak-memory logging, since those two numbers are the actual headline of
Phase 2's comparison story (reports/phase2_comparison.md), not just an
AUROC number.
"""

import argparse
from pathlib import Path

import torch
import wandb
import yaml
from torch.utils.data import DataLoader

from src.data.chestxray_dataset import ChestXrayDataset
from src.data.split_utils import load_split_list, split_train_val
from src.data.vit_transforms import get_vit_eval_transforms, get_vit_train_transforms
from src.models.vit_classifier import ViTClassifier
from src.training.train import compute_class_weights, train_one_epoch, validate
from src.training.utils import get_device, save_checkpoint, set_seed
from src.training.vit_optim import build_cosine_scheduler, build_differential_lr_optimizer


def count_trainable_params(model: torch.nn.Module) -> tuple[int, int]:
    """
    Returns (trainable_params, total_params). Full fine-tune should show
    trainable == total (nothing frozen); comparing this number against
    train_vit_lora.py's run is the whole point of logging it (Phase 2's
    success criteria include a trainable-param comparison, not just AUROC).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 ViT-Base — full fine-tune")
    parser.add_argument("--config", type=str, default="configs/vit_full.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]
    classes = config["classes"]

    set_seed(train_cfg["seed"])
    device = get_device()

    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=config["wandb"]["run_name"],
        tags=config["wandb"]["tags"],
        config=config,
    )

    # --- Data ---
    # Same split_utils.py / ChestXrayDataset as Phase 1 — only the transform
    # module differs (vit_transforms, not transforms, for ViT's own
    # normalization stats; see src/data/vit_transforms.py's docstring).
    train_val_filenames = load_split_list(f"{data_cfg['data_dir']}/{data_cfg['train_val_list']}")
    train_filenames, val_filenames = split_train_val(
        train_val_filenames,
        labels_csv_path=f"{data_cfg['data_dir']}/{data_cfg['labels_csv']}",
        val_fraction=data_cfg["val_fraction"],
        seed=train_cfg["seed"],
    )

    train_dataset = ChestXrayDataset(
        data_dir=data_cfg["data_dir"],
        image_filenames=train_filenames,
        labels_csv_path=f"{data_cfg['data_dir']}/{data_cfg['labels_csv']}",
        classes=classes,
        transform=get_vit_train_transforms(data_cfg["image_size"]),
    )
    val_dataset = ChestXrayDataset(
        data_dir=data_cfg["data_dir"],
        image_filenames=val_filenames,
        labels_csv_path=f"{data_cfg['data_dir']}/{data_cfg['labels_csv']}",
        classes=classes,
        transform=get_vit_eval_transforms(data_cfg["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"]
    )

    # --- Model, loss, optimizer ---
    model = ViTClassifier(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    trainable_params, total_params = count_trainable_params(model)
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
    if config["wandb"]["log_trainable_params"]:
        wandb.log({"params/trainable": trainable_params, "params/total": total_params})

    pos_weight = compute_class_weights(train_dataset, classes).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Differential LR: backbone_learning_rate << head_learning_rate — see
    # src/training/vit_optim.py's docstring for why full fine-tune needs
    # this and LoRA doesn't. This call raises NotImplementedError until the
    # user implements build_differential_lr_optimizer().
    optimizer = build_differential_lr_optimizer(
        model,
        backbone_lr=train_cfg["backbone_learning_rate"],
        head_lr=train_cfg["head_learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_cosine_scheduler(optimizer, epochs=train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_cfg["mixed_precision"]))

    # --- Training loop ---
    # Identical structure to Phase 1's train.py main() — same checkpoint-on-
    # best-val-AUROC / early-stopping logic, just against vit_full.yaml's
    # tighter early_stopping_patience (3, not 5).
    best_mean_auroc = 0.0
    epochs_without_improvement = 0
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, train_cfg["log_every_n_steps"]
        )
        val_loss, mean_auroc, per_class_auroc = validate(model, val_loader, criterion, classes, device, epoch)
        scheduler.step()

        log_dict = {
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "val/epoch_loss": val_loss,
            "val/mean_auroc": mean_auroc,
            # Two LRs, not one — Phase 2's comparison needs both visible,
            # not just "a schedule was used" (vit_optim.py Interview Notes Q1).
            "train/backbone_lr": scheduler.get_last_lr()[0],
            "train/head_lr": scheduler.get_last_lr()[1],
        }
        log_dict.update({f"val/auroc_{name}": score for name, score in per_class_auroc.items()})
        if config["wandb"]["log_gpu_memory"] and device.type == "cuda":
            log_dict["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            log_dict["gpu/memory_peak_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        wandb.log(log_dict)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} mean_auroc={mean_auroc:.4f}")

        if mean_auroc > best_mean_auroc:
            best_mean_auroc = mean_auroc
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                {"mean_auroc": mean_auroc, "per_class_auroc": per_class_auroc},
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= train_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no improvement for {epochs_without_improvement} epochs)")
                break

    wandb.finish()


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why reuse train_one_epoch/validate from src/training/train.py instead
#     of writing ViT-specific versions?
# A1: Those functions only call model(images) and criterion(logits, labels) —
#     nothing ResNet-specific. Keeping the loop identical across Phase 1 and
#     2 is deliberate: it means every measured difference between ResNet-50,
#     ViT full fine-tune, and ViT LoRA comes from the model/optimizer, not
#     from subtly different training code each time.
#
# Q2: Why log GPU memory PEAK (max_memory_allocated) here when Phase 1 only
#     logged current allocated memory?
# A2: Current allocated memory at the moment of logging (end of
#     epoch/validation) understates the actual peak during training — e.g.
#     the backward pass briefly holds activations for the whole batch plus
#     gradients. Peak memory is what determines whether a given batch size
#     fits on a given GPU, which is exactly the number Phase 2's hardware
#     comparison (RTX 3050 LoRA vs Kaggle T4 full fine-tune) needs.
#
# Q3: This script will raise NotImplementedError today — why scaffold a
#     training script around a function that doesn't work yet?
# A3: The stub methodology (PROJECT_BLUEPRINT.md Section 3.1) is about the
#     user implementing the 1-3 functions that matter most for understanding
#     the concept — build_differential_lr_optimizer here — while everything
#     around it (data loading, the training loop, logging, checkpointing) is
#     already correct and doesn't need to be rebuilt from scratch just to
#     practice the one new idea this phase introduces.
