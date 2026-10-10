"""
Loads and further splits the NIH ChestX-ray14 official patient-level splits.

Why this file exists: the NIH release ships two split files — train_val_list.txt
and test_list.txt — that guarantee no patient appears in both train_val and test
(this is the "official patient-level split" PROJECT_BLUEPRINT.md Section 5.1
mandates; random splitting here would leak a patient's other X-rays across
train/test and inflate reported AUROC, which is exactly how some published
chest X-ray papers were later found to be overly optimistic).

train_val_list.txt itself is NOT pre-split into train/val, so we do one more
split here — but we must do it at the PATIENT level (via "Patient ID" in
Data_Entry_2017.csv), not the image level, for the same leakage reason: a
single patient can have many follow-up X-rays in the dataset.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_split_list(split_file: Path) -> list[str]:
    """Read a NIH split file (one image filename per line) into a list."""
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def split_train_val(
    train_val_filenames: list[str],
    labels_csv_path: Path,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """
    Split the combined train_val image list into train/val, grouped by patient.

    Args:
        train_val_filenames: image filenames from train_val_list.txt
        labels_csv_path: path to Data_Entry_2017.csv (used only for its
            "Image Index" -> "Patient ID" mapping, not for labels)
        val_fraction: fraction of PATIENTS (not images) to hold out for validation
        seed: random seed, so the split is reproducible across runs

    Returns:
        (train_filenames, val_filenames)
    """
    labels_df = pd.read_csv(labels_csv_path)
    patient_map = labels_df.set_index("Image Index")["Patient ID"]

    subset = pd.DataFrame({"filename": train_val_filenames})
    subset["patient_id"] = subset["filename"].map(patient_map)

    # GroupShuffleSplit ensures every image from a given patient lands in the
    # same side of the split — this is the patient-level guarantee from
    # Section 5.1, applied to our own train/val carve-out of train_val_list.txt.
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(subset, groups=subset["patient_id"]))

    train_filenames = subset.iloc[train_idx]["filename"].tolist()
    val_filenames = subset.iloc[val_idx]["filename"].tolist()
    return train_filenames, val_filenames


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why not just use sklearn's train_test_split with a fixed random_state?
# A1: Because it splits at the image level. NIH ChestX-ray14 has ~4 follow-up
#     images per patient on average; a plain random split would put some of a
#     patient's X-rays in train and others in val/test, so the model could
#     partly "memorize" patient-specific anatomy rather than learn disease
#     signal. GroupShuffleSplit keeps every image for a given Patient ID on
#     one side of the split.
#
# Q2: The NIH release already gives you train_val_list.txt and test_list.txt —
#     why do you need to split again?
# A2: Those two files only separate (train+val) from test. There's no NIH-
#     provided train/val boundary inside train_val_list.txt, so we still have
#     to carve out a validation set ourselves — and we must do that carve-out
#     at the same patient-level granularity, not randomly by image, for the
#     same leakage reason as Q1.
#
# Q3: What would happen if you ignored patient grouping entirely?
# A3: Reported validation/test AUROC would likely be optimistically biased —
#     the model could partially fit to patient-specific texture/positioning
#     rather than true pathology, and that inflated number would not
#     reproduce on genuinely unseen patients in deployment.
