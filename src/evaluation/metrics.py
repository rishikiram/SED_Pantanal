import numpy as np
from sklearn.metrics import f1_score


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
    # Only score classes that appear in targets to avoid undefined F1
    active = targets.sum(axis=0) > 0
    if active.sum() == 0:
        return 0.0
    return f1_score(targets[:, active], preds[:, active], average='macro', zero_division=0)
