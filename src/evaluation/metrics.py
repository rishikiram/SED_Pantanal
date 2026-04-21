import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def macro_roc_auc(
    probs: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Macro-averaged ROC-AUC, skipping classes with no positive labels.

    probs:   (N, num_classes) — sigmoid probabilities
    targets: (N, num_classes) — binary labels
    """
    targets = (targets > 0).astype(int)
    active = targets.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0
    return float(roc_auc_score(targets[:, active], probs[:, active], average='macro'))


def segment_f1(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Macro-averaged segment F1 across all classes.

    probs:   (N, num_classes) — sigmoid probabilities
    targets: (N, num_classes) — binary labels
    """
    preds = (probs >= threshold).astype(int)
    targets = (targets > 0).astype(int)  # binarize: secondary labels (0.5) count as positive
    # Only score classes that appear in targets to avoid undefined F1
    active = targets.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0
    return f1_score(targets[:, active], preds[:, active], average='macro', zero_division=0)
