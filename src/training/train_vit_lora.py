"""
Phase 2 training loop: ViT-Base LoRA fine-tune on NIH ChestX-ray14.

Why this file exists: the other half of Phase 2's full-vs-LoRA comparison
(train_vit_full.py is the full fine-tune arm). Reuses the exact same
train_one_epoch/validate/compute_class_weights from Phase 1's train.py, the
same ViTClassifier, and the same ViT-specific transforms as
train_vit_full.py — the only things that differ are: apply_lora_to_vit() is
called before training (wrapping the backbone in frozen weights + trainable
low-rank adapters), a single flat learning rate is used instead of
differential LR (no pretrained-vs-random-init asymmetry left to protect —
see src/training/vit_optim.py's docstring), and the checkpoint saved at the
end is the PEFT adapter alone, not a full model state_dict — that's the
entire point of parameter-efficient fine-tuning: a few hundred KB of
adapter weights instead of ~86M params' worth of a full checkpoint.
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
from src.models.vit_lora import apply_lora_to_vit
from src.training.train import compute_class_weights, train_one_epoch, validate
from src.training.train_vit_full import count_trainable_params
from src.training.utils import get_device, set_seed
from src.training.vit_optim import build_cosine_scheduler


def save_lora_checkpoint(path: str | Path, model: torch.nn.Module, epoch: int, metrics: dict) -> None:
    """
    Save the PEFT adapter only (model.save_pretrained), plus the metrics
    that produced it, separately from any base-model weights.

    Why not reuse src/training/utils.py's save_checkpoint: that function
    calls model.state_dict(), which for a peft-wrapped model includes every
    frozen backbone parameter too — correct but wasteful (defeats LoRA's
    storage advantage) and conflates "the adapter" with "the whole model."
    Keeping the adapter checkpoint separate from the frozen base model
    mirrors how it would actually be deployed: ship the ~86M-param base
    once, swap small adapter files for different fine-tunes of it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)  # peft.PeftModel.save_pretrained — adapter weights + config only
    torch.save({"epoch": epoch, "metrics": metrics}, path / "training_metrics.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 ViT-Base — LoRA fine-tune")
    parser.add_argument("--config", type=str, default="configs/vit_lora.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_cfg = config["training"]
    lora_cfg = config["lora"]
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

    # --- Data (identical to train_vit_full.py) ---
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

    # --- Model + LoRA ---
    model = ViTClassifier(
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
        pretrained=config["model"]["pretrained"],
    )
    # Wraps model.backbone's query/value projections with trainable LoRA
    # adapters; everything else (base backbone weights) frozen. Raises
    # NotImplementedError until the user implements apply_lora_to_vit().
    model = apply_lora_to_vit(
        model,
        rank=lora_cfg["rank"],
        alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        dropout=lora_cfg["dropout"],
    )
    model = model.to(device)

    trainable_params, total_params = count_trainable_params(model)
    print(
        f"Trainable params: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%) — compare against train_vit_full.py's run"
    )
    if config["wandb"]["log_trainable_params"]:
        wandb.log({"params/trainable": trainable_params, "params/total": total_params})

    pos_weight = compute_class_weights(train_dataset, classes).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Single flat LR across every trainable param (adapters + head) — unlike
    # train_vit_full.py's differential LR, there's no pretrained-weight risk
    # to protect against here (see src/training/vit_optim.py docstring).
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_cosine_scheduler(optimizer, epochs=train_cfg["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and train_cfg["mixed_precision"]))

    # --- Training loop ---
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
            "lr": scheduler.get_last_lr()[0],
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
            save_lora_checkpoint(
                checkpoint_dir / "best_adapter",
                model,
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
# Q1: Why save the LoRA adapter separately (save_pretrained) instead of a
#     full model state_dict like Phase 1 and train_vit_full.py do?
# A1: That's the deployment-relevant benefit of LoRA, not just a training-
#     time one: a full ViT-Base checkpoint is ~330MB (86M params x 4 bytes);
#     a rank-8 adapter over query+value projections is a few hundred KB.
#     Saving only the adapter means the frozen base model is downloaded/
#     stored once, and different fine-tunes of it (different diseases,
#     different datasets) ship as small adapter files layered on top.
#
# Q2: Why filter(lambda p: p.requires_grad, model.parameters()) for the
#     optimizer instead of just model.parameters()?
# A2: After apply_lora_to_vit(), most backbone parameters have
#     requires_grad=False (frozen). Passing them to AdamW anyway wouldn't
#     break training (frozen params never receive gradients, so they'd never
#     update) but would waste memory on optimizer state (momentum + variance
#     estimates) for every frozen parameter — exactly the memory LoRA is
#     supposed to save in the first place.
#
# Q3: Both train_vit_full.py and this script early-stop on validation mean
#     AUROC with the same patience (3) — why keep that identical given LoRA
#     and full fine-tune are different training regimes?
# A3: Patience controls how long to wait for improvement before concluding
#     training has plateaued, not how fast a model learns — keeping it
#     identical across both arms means neither gets an "unfair" extra
#     chance to improve, which matters for a controlled comparison (the same
#     reason train_one_epoch/validate/compute_class_weights are reused
#     unchanged rather than re-tuned per arm).
