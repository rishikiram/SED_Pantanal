import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCELoss(nn.Module):
    """Focal BCE loss with per-class weighting.

    logits: (B, T', num_classes) — frame-level model output
    targets: (B, num_classes)   — window-level labels, broadcast across T'

    The window-level label is applied to every frame (weak supervision).
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits:  (B, T', C)
        targets: (B, C)  — broadcast to (B, T', C)
        """
        # Broadcast targets across time frames
        targets = targets.unsqueeze(1).expand_as(logits)   # (B, T', C)

        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Focal weighting
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if class_weights is not None:
            loss = loss * class_weights.to(logits.device)

        return loss.mean()


def compute_class_weights(label_counts: torch.Tensor) -> torch.Tensor:
    """Inverse-sqrt frequency class weights.

    label_counts: (num_classes,) — number of positive samples per class
    Returns: (num_classes,) weights
    """
    n = label_counts.sum()
    num_classes = label_counts.shape[0]
    weights = (n / (num_classes * label_counts.clamp(min=1))) ** 0.5
    return weights
