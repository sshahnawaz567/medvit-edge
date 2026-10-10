# MedViT-Edge

Fine-tuned Vision Transformer for chest X-ray classification, with a full
training → optimization → deployment pipeline: ResNet-50 and ViT-Base
fine-tuning (full vs LoRA) on NIH ChestX-ray14, quantization (FP32 → FP16 →
INT8), TensorRT benchmarking, and deployment to Hugging Face Spaces and
Microsoft Azure.

See [`PROJECT_BLUEPRINT.md`](./PROJECT_BLUEPRINT.md) for the full project plan,
phase breakdown, and working agreement — it is the single source of truth.

**Status:** Phase 1 (ResNet-50 baseline) in progress.

## Dataset Setup

This project uses [NIH ChestX-ray14](https://www.kaggle.com/datasets/nih-chest-xrays/data)
(112,120 images, 30,805 patients, ~45GB, public domain).

1. Download via the Kaggle CLI (requires a Kaggle account with phone verification):
   ```
   kaggle datasets download -d nih-chest-xrays/data -p data/nih-chestxray --unzip
   ```
2. Confirm the extracted layout matches what `configs/baseline.yaml` expects:
   ```
   data/nih-chestxray/
     images_001/images/*.png
     images_002/images/*.png
     ...
     images_012/images/*.png
     Data_Entry_2017.csv
     train_val_list.txt
     test_list.txt
   ```
3. **Never** create your own random train/test split — always use the official
   `train_val_list.txt` / `test_list.txt` patient-level split files (see
   `PROJECT_BLUEPRINT.md` Section 5.1 and `src/data/split_utils.py`).

## Environment Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Phase 1: ResNet-50 Baseline

```powershell
python -m src.training.train --config configs/baseline.yaml
python -m src.evaluation.evaluate --config configs/baseline.yaml --checkpoint checkpoints/phase1_resnet50/best.pt
```

Training logs every metric to Weights & Biases under project `medvit-edge`.
