# Phase 1 Learnings: ResNet-50 Baseline on NIH ChestX-ray14

## Key Discovery: Overfitting + Best Checkpoint Rescue

The model overfit after epoch 4. Validation loss climbed from 0.90 → 1.45 while 
training loss kept dropping — classic signature. However, because the code saved 
"best checkpoint on validation metric", my deployed model is from **epoch 4 with 
mean AUROC 0.8258**, not the degraded epoch 9 (0.80).

### Verification
Loaded checkpoint programmatically:
- epoch: 4
- metrics.mean_auroc: 0.8258

### Resume lesson
"Always save best checkpoint on validation metric, not last epoch." Without this, 
I would have deployed a worse model while reporting a worse number — both wrong.

### Phase 2 plan
Reduce early_stopping_patience from 5 to 2-3. Add stronger regularization 
(dropout 0.3, weight_decay 1e-4). Consider mixup augmentation.