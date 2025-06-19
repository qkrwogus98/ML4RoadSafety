import torch
import torch.nn.functional as F

def focal_loss(input, target, gamma=2.0, weight=None):
    """Compute focal loss for binary classification."""
    bce = F.binary_cross_entropy(input, target, reduction="none", weight=weight)
    p_t = input * target + (1 - input) * (1 - target)
    loss = (1 - p_t) ** gamma * bce
    return loss.mean()
