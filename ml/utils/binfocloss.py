import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss implementation for addressing extreme class imbalance.
    
    Down-weights well-classified (easy) examples and forces the model to focus 
    on hard, misclassified examples during training.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initializes the BinaryFocalLoss module.

        Args:
            alpha (float): Class balancing factor. Increase above 0.5 if the positive class is rare.
            gamma (float): Focusing parameter. Higher values suppress loss for confident predictions.
            reduction (str): Specifies the reduction to apply: 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the focal loss between predicted logits and targets.

        Args:
            logits (torch.Tensor): Unnormalized model predictions.
            targets (torch.Tensor): Ground truth binary labels matching logits shape.

        Returns:
            torch.Tensor: Scaled loss tensor according to the chosen reduction.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        
        # Calculate p_t (the model's estimated probability for the correct ground truth class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Compute modulating factor and class-balancing weight
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss