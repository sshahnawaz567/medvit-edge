"""
Per-class and mean AUROC for multi-label chest X-ray classification.

Why this file exists: AUROC (not accuracy) is the standard metric for this
task — ChestX-ray14 is severely imbalanced per class (Hernia <0.2%,
Infiltration ~17%, per PROJECT_BLUEPRINT.md Section 5.1), so a model that
always predicts "negative" would score >99% accuracy on Hernia while being
useless. AUROC is threshold-independent and far less distorted by this
imbalance, which is also why CheXNet and nearly every ChestX-ray14 paper
report AUROC rather than accuracy.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_per_class_auroc(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict[str, float]:
    """
    Compute AUROC independently for each of the 14 disease classes.

    Args:
        y_true: shape (n_samples, n_classes), multi-hot ground truth (0/1).
        y_pred: shape (n_samples, n_classes), predicted probabilities (post-sigmoid,
            NOT raw logits — sklearn's roc_auc_score expects scores where higher
            means "more positive", which logits also satisfy, but probabilities
            are the convention used elsewhere in this project).
        classes: class names in the same column order as y_true/y_pred.

    Returns:
        dict mapping each class name to its AUROC score, or float('nan') for
        any class where AUROC is undefined (see hint 2).
    """
    per_class_auroc: dict[str, float] = {}
    for i, class_name in enumerate(classes):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
        except ValueError:
            # Only one label value present in y_true[:, i]
            score = float("nan")
        per_class_auroc[class_name] = score
    return per_class_auroc


def compute_mean_auroc(per_class_auroc: dict[str, float]) -> float:
    """
    Average per-class AUROC scores into a single headline number.

    Args:
        per_class_auroc: output of compute_per_class_auroc — may contain
            float('nan') entries for classes where AUROC was undefined.

    Returns:
        Mean AUROC across classes, ignoring NaN entries.
    """
    scores = list(per_class_auroc.values())
    if all(np.isnan(s) for s in scores):
        import warnings
        warnings.warn("All per-class AUROC values are NaN — check y_true/y_pred shape and class order.")
    return float(np.nanmean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Interview Notes
# ─────────────────────────────────────────────────────────────────────────────
# Q1: Why AUROC instead of accuracy or F1 for this task?
# A1: Severe per-class imbalance (Hernia <0.2% prevalence) makes accuracy
#     trivially gameable by always predicting negative. AUROC measures ranking
#     quality across all thresholds and is far more robust to class imbalance,
#     which is why it's the standard metric in the ChestX-ray14 literature
#     (CheXNet et al.) that this baseline is calibrated against.
#
# Q2: Why can AUROC be undefined for a class, and what should happen when it is?
# A2: AUROC requires at least one positive and one negative example in
#     y_true to be computable — if a given eval split happens to contain zero
#     positive cases for a rare class (plausible for Hernia in a small
#     validation split), the metric has no mathematical definition. We record
#     NaN for that class rather than crashing the whole evaluation, and
#     exclude it from the mean via nanmean rather than treating it as 0.
#
# Q3: Why does mean AUROC matter more than any single class's AUROC for this
#     project's "success criteria" (Section 7, Phase 1: mean AUROC >= 0.80)?
# A3: It's a single comparable headline number across phases/models (ResNet-50
#     baseline vs ViT full fine-tune vs LoRA), but the per-class breakdown is
#     still reported alongside it (reports/) because a model can hit a strong
#     mean while quietly failing on a specific disease — worth surfacing for
#     clinical credibility even though it isn't the single pass/fail gate.
