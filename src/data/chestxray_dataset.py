"""
PyTorch Dataset for NIH ChestX-ray14 multi-label classification.

Why this file exists: wraps the raw NIH release (per-folder PNGs +
Data_Entry_2017.csv) into a standard torch Dataset that train.py and
evaluate.py can hand to a DataLoader. Image-path resolution and split-file
reading are provided; __getitem__ and _load_labels are left for the user to
implement (PROJECT_BLUEPRINT.md Section 7, Phase 1 stubs) since they're the
two places where "the data pipeline is actually correct" lives — get the
label encoding or the image loading wrong and every downstream metric is
silently meaningless.
"""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        image_filenames: list[str],
        labels_csv_path: str | Path,
        classes: list[str],
        transform=None,
    ):
        """
        Args:
            data_dir: root folder of the NIH download, expected to contain
                images_001/images/, images_002/images/, ..., images_012/images/.
            image_filenames: the specific image filenames belonging to this
                split (train, val, or test) — produced by split_utils.py,
                NOT by scanning the directory, so patient-level split
                guarantees from split_utils.py are respected here.
            labels_csv_path: path to Data_Entry_2017.csv.
            classes: fixed 14-class order from configs/baseline.yaml — every
                label vector this dataset returns uses this exact order.
            transform: torchvision transform from transforms.py, applied in
                __getitem__.
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_dir}. Download NIH ChestX-ray14 from "
                "Kaggle (nih-chest-xrays/data) and extract it to this path — see README.md "
                "'Dataset Setup' for the exact steps. (PROJECT_BLUEPRINT.md Section 5.)"
            )

        self.image_filenames = image_filenames
        self.classes = classes
        self.transform = transform

        # Image files are spread across 12 subfolders (images_001..images_012),
        # each with its own nested images/ directory. We index them once here
        # (not per __getitem__ call) so lookups during training are O(1) dict
        # access instead of repeated globbing.
        self.image_paths = self._build_image_index()

        # STUB: builds {filename: multi-hot label tensor} — see below.
        self.labels = self._load_labels(Path(labels_csv_path))

    def _build_image_index(self) -> dict[str, Path]:
        """Map every image filename under data_dir/images_*/images/ to its full path."""
        index: dict[str, Path] = {}
        for image_path in self.data_dir.glob("images_*/images/*.png"):
            index[image_path.name] = image_path
        missing = [f for f in self.image_filenames if f not in index]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) from the split list were not found under "
                f"{self.data_dir}/images_*/images/ — first missing: {missing[0]}. "
                "Check that the full NIH archive was extracted (all 12 images_* folders)."
            )
        return index

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load one (image, label) pair by index.

        Returns:
            image: transformed image tensor, shape (3, H, W)
            label: multi-hot float tensor, shape (num_classes,) — 1.0 for each
                disease present, all-zero for "No Finding"
        """
        filename = self.image_filenames[idx]
        image_path = self.image_paths[filename]

        image = Image.open(image_path).convert("L")
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[filename]
        return image, label

    def _load_labels(self, labels_csv_path: Path) -> dict[str, torch.Tensor]:
        """
        Parse Data_Entry_2017.csv into {filename: multi-hot label tensor}.

        The CSV's "Finding Labels" column contains pipe-separated disease
        names (e.g. "Cardiomegaly|Effusion") or the literal string
        "No Finding". Only rows for filenames in self.image_filenames need
        labels — this dataset instance only ever serves one split.

        Returns:
            dict mapping each filename in self.image_filenames to a float
            tensor of shape (len(self.classes),), using self.classes for the
            index order. "No Finding" -> all-zero vector.
        """
        df = pd.read_csv(labels_csv_path)
        wanted = set(self.image_filenames)
        df = df[df["Image Index"].isin(wanted)]

        name_to_idx = {name: i for i, name in enumerate(self.classes)}
        num_classes = len(self.classes)

        labels: dict[str, torch.Tensor] = {}
        for _, row in df.iterrows():
            filename = row["Image Index"]
            vec = torch.zeros(num_classes, dtype=torch.float32)
            for finding in row["Finding Labels"].split("|"):
                if finding != "No Finding":
                    vec[name_to_idx[finding]] = 1.0
            labels[filename] = vec

        missing = [f for f in self.image_filenames if f not in labels]
        if missing:
            raise ValueError(
                f"{len(missing)} filenames have no matching row in {labels_csv_path} "
                f"— first missing: {missing[0]}"
            )

        return labels


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why build the image-path index once in __init__ instead of resolving
#     each path inside __getitem__?
# A1: __getitem__ is called once per sample per epoch — with 112K images that's
#     112K filesystem globs per epoch if done lazily. Building a filename->Path
#     dict once up front turns every lookup into O(1) dict access, which
#     matters a lot once DataLoader workers are hammering this in parallel.
#
# Q2: This is a multi-label problem (an image can have multiple diseases) —
#     how does that change the dataset/label design compared to multi-class?
# A2: Labels are a multi-hot float vector (can contain multiple 1s, or be all
#     zero for "No Finding"), not a single class index. That decision ripples
#     downstream: train.py must use BCEWithLogitsLoss (one independent binary
#     decision per class) instead of CrossEntropyLoss, and metrics.py computes
#     per-class AUROC rather than a single multi-class accuracy.
#
# Q3: Why raise FileNotFoundError eagerly in __init__ rather than letting it
#     fail later inside __getitem__ during training?
# A3: Failing fast at dataset construction (before any DataLoader workers spin
#     up or any W&B run starts) gives an immediate, actionable error pointing
#     at the README download steps, instead of a confusing worker-process
#     traceback minutes into a training run.
