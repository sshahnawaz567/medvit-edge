"""
Runs a trained checkpoint over the held-out test set and reports AUROC.

Why this file exists: separates "did training converge" (train.py's per-epoch
validation) from "what's the final, reportable number on data the model never
saw during training or model selection" — the test set (test_list.txt) is
only ever touched here, once, after training is finished.
"""

import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.chestxray_dataset import ChestXrayDataset
from src.data.split_utils import load_split_list
from src.data.transforms import get_eval_transforms
from src.evaluation.metrics import compute_mean_auroc, compute_per_class_auroc
from src.models.baseline_resnet import BaselineResNet
from src.training.utils import get_device, load_checkpoint


def run_evaluation(config: dict, checkpoint_path: str) -> dict:
    device = get_device()

    data_cfg = config["data"]
    classes = config["classes"]

    test_filenames = load_split_list(f"{data_cfg['data_dir']}/{data_cfg['test_list']}")
    test_dataset = ChestXrayDataset(
        data_dir=data_cfg["data_dir"],
        image_filenames=test_filenames,
        labels_csv_path=f"{data_cfg['data_dir']}/{data_cfg['labels_csv']}",
        classes=classes,
        transform=get_eval_transforms(data_cfg["image_size"]),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
    )

    model = BaselineResNet(num_classes=config["model"]["num_classes"], pretrained=False)
    load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_probs, axis=0)

    per_class = compute_per_class_auroc(y_true, y_pred, classes)
    mean_auroc = compute_mean_auroc(per_class)

    print("\nPer-class AUROC (test set):")
    for class_name, score in per_class.items():
        print(f"  {class_name:<20s} {score:.4f}")
    print(f"\nMean AUROC: {mean_auroc:.4f}")

    return {"per_class_auroc": per_class, "mean_auroc": mean_auroc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Phase 1 checkpoint on the NIH test split")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_evaluation(config, args.checkpoint)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why is this a separate script from train.py's validation loop rather
#     than just reusing the same validate() function on the test set?
# A1: They answer different questions. Validation (inside train.py) runs every
#     epoch and is used for model selection (checkpointing, early stopping) —
#     that makes the validation set influence training decisions. The test
#     set must stay completely unused until after the model is finalized, or
#     the reported number stops being an honest estimate of generalization.
#     Keeping evaluate.py separate makes that boundary structurally explicit.
#
# Q2: Why torch.no_grad() around the inference loop?
# A2: Disables autograd's computation-graph tracking, which isn't needed for
#     inference and would otherwise waste GPU memory and compute holding onto
#     activations for a backward pass that will never run.
#
# Q3: Why pretrained=False when constructing the model here, given training
#     used pretrained=True?
# A3: load_checkpoint() immediately overwrites every weight with the trained
#     checkpoint's state_dict, so downloading/loading ImageNet weights first
#     would be pure wasted work (and an unnecessary network call) before
#     they're discarded.
