# Phase 2: ViT-Base Full Fine-Tune vs LoRA — Comparison

> Filled in by the user after both training runs complete. See
> PROJECT_BLUEPRINT.md Section 7 (Phase 2 success criteria) for targets.

## Headline Results

| Metric | ResNet-50 Baseline (Phase 1) | ViT-Base Full Fine-Tune | ViT-Base LoRA (rank=8) |
|---|---|---|---|
| Mean AUROC (val) | 0.826 (epoch 4) | | |
| Mean AUROC (test) | | | |
| Trainable parameters | | | |
| Total parameters | | | |
| Trainable % | | | |
| Peak GPU memory (training) | | | |
| Training hardware | Kaggle T4 x2 | | |
| Epochs to best checkpoint | | | |
| Wall-clock training time | | | |

## Per-Class AUROC

| Class | ResNet-50 | ViT Full | ViT LoRA |
|---|---|---|---|
| Atelectasis | | | |
| Cardiomegaly | | | |
| Effusion | | | |
| Infiltration | | | |
| Mass | | | |
| Nodule | | | |
| Pneumonia | | | |
| Pneumothorax | | | |
| Consolidation | | | |
| Edema | | | |
| Emphysema | | | |
| Fibrosis | | | |
| Pleural_Thickening | | | |
| Hernia | | | |

## Convergence Curves

<!-- Link or embed W&B charts: val/mean_auroc vs epoch, train/epoch_loss vs epoch,
     for all three runs on the same axes. -->

## Discussion

<!-- Did LoRA land within 2% mean AUROC of full fine-tune (Phase 2 success
     criterion)? What's the memory/parameter tradeoff actually worth in
     practice? Anything surprising relative to expectations going in? -->

## W&B Run Links

- ResNet-50 baseline:
- ViT-Base full fine-tune:
- ViT-Base LoRA (rank=8):
