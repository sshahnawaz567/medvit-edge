"""
Phase 1 training loop: ResNet-50 baseline on NIH ChestX-ray14, with W&B logging.

Why this file exists: this is the reproducible entry point for Phase 1 — every
hyperparameter comes from configs/baseline.yaml (no magic numbers here), every
run is logged to Weights & Biases (PROJECT_BLUEPRINT.md Section 10.7: "Always
log to W&B. Every training run."), and the best checkpoint by validation mean
AUROC is what evaluate.py later scores on the held-out test set.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.chestxray_dataset import ChestXrayDataset
from src.data.split_utils import load_split_list, split_train_val
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.evaluation.metrics import compute_mean_auroc, compute_per_class_auroc
from src.models.baseline_resnet import BaselineResNet
from src.training.utils import get_device, save_checkpoint, set_seed


def compute_class_weights(dataset: ChestXrayDataset, classes: list[str]) -> torch.Tensor:
    """
    Compute per-class pos_weight for nn.BCEWithLogitsLoss to counter the
    severe class imbalance in ChestX-ray14 (Hernia <0.2% vs Infiltration ~17%,
    per PROJECT_BLUEPRINT.md Section 5.1). Without this, the loss is dominated
    by common classes and the model has little incentive to learn rare ones.

    Args:
        dataset: a ChestXrayDataset (typically the training split) whose
            .labels dict holds the multi-hot label tensor for every sample.
        classes: class names, same order as the label vectors.

    Returns:
        1-D float tensor of shape (len(classes),), passed as
        nn.BCEWithLogitsLoss(pos_weight=...). Each entry should be
        (# negative examples / # positive examples) for that class — values
        > 1 upweight the (rarer) positive class in the loss.
    """
    all_labels = torch.stack(list(dataset.labels.values()))
    num_samples = all_labels.shape[0]

    num_pos = all_labels.sum(dim=0)
    num_neg = num_samples - num_pos

    pos_weight = num_neg / num_pos.clamp(min=1)
    pos_weight = pos_weight.clamp(min=1.0, max=50.0)

    return pos_weight


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch, log_every_n_steps):
    model.train()
    running_loss = 0.0

    for step, (images, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch} [train]")):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        # Mixed precision (autocast + GradScaler) is required to fit
        # batch_size=16 ResNet-50 training in the RTX 3050's 4GB VRAM —
        # see PROJECT_BLUEPRINT.md Section 7 hardware context.
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if step % log_every_n_steps == 0:
            wandb.log({"train/step_loss": loss.item(), "epoch": epoch})

    return running_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, classes, device, epoch):
    model.eval()
    running_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in tqdm(loader, desc=f"Epoch {epoch} [val]"):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()

        all_labels.append(labels.cpu().numpy())
        all_probs.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)

    per_class_auroc = compute_per_class_auroc(y_true, y_pred, classes)
    mean_auroc = compute_mean_auroc(per_class_auroc)
    val_loss = running_loss / len(loader)

    return val_loss, mean_auroc, per_class_auroc


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 ResNet-50 baseline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
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
        config=config,  # log the full config as hyperparameters, per Section 10.7
    )

    # --- Data ---
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
        transform=get_train_transforms(data_cfg["image_size"]),
    )
    val_dataset = ChestXrayDataset(
        data_dir=data_cfg["data_dir"],
        image_filenames=val_filenames,
        labels_csv_path=f"{data_cfg['data_dir']}/{data_cfg['labels_csv']}",
        classes=classes,
        transform=get_eval_transforms(data_cfg["image_size"]),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=data_cfg["num_workers"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=data_cfg["num_workers"]
    )

    # --- Model, loss, optimizer ---
    model = BaselineResNet(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)

    pos_weight = compute_class_weights(train_dataset, classes).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=train_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_cfg["mixed_precision"]))

    # --- Training loop ---
    best_mean_auroc = 0.0
    epochs_without_improvement = 0
    checkpoint_dir = Path(train_cfg["checkpoint_dir"])

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
            "lr": scheduler.get_last_lr()[0],
        }
        log_dict.update({f"val/auroc_{name}": score for name, score in per_class_auroc.items()})
        if config["wandb"]["log_gpu_memory"] and device.type == "cuda":
            log_dict["gpu/memory_allocated_mb"] = torch.cuda.memory_allocated(device) / (1024**2)
            log_dict["gpu/memory_reserved_mb"] = torch.cuda.memory_reserved(device) / (1024**2)
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
# Q1: Why BCEWithLogitsLoss with a pos_weight instead of, say, focal loss,
#     given how imbalanced ChestX-ray14 is?
# A1: pos_weight-weighted BCE is the standard, well-understood first baseline
#     for multi-label imbalance and keeps Phase 1 focused on establishing a
#     reproducible CheXNet-comparable number with a simple, explainable loss.
#     Focal loss (or other imbalance-specific losses) is a reasonable later
#     experiment, but introducing it in the baseline would conflate "is the
#     pipeline correct" with "is this a better loss function" — two different
#     questions this project wants to answer separately.
#
# Q2: Why checkpoint on validation mean AUROC rather than validation loss?
# A2: Loss and AUROC can diverge under class imbalance — a model can keep
#     lowering BCE loss by getting more confident on easy, common classes
#     while rare-class ranking quality stagnates or worsens. Mean AUROC is
#     this project's actual success metric (Section 7: "Mean AUROC >= 0.80"),
#     so model selection should optimize for that directly, not a proxy.
#
# Q3: Why CosineAnnealingLR instead of a step schedule or constant LR?
# A3: Cosine annealing smoothly decays the learning rate to near-zero by the
#     final epoch without needing hand-tuned step milestones, which is a
#     reasonable default when there isn't yet strong evidence (from prior
#     runs on this exact dataset/model) that a different schedule shape would
#     do meaningfully better — a sensible choice to revisit if val_loss curves
#     show restarting training was premature.
