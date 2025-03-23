import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss3D(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss3D, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Compute Dice Loss for multi-class 3D segmentation.

        Args:
            pred (torch.Tensor): Predicted logits (batch_size, num_classes, D, H, W)
            target (torch.Tensor): Ground truth class indices (batch_size, 1, D, H, W)

        Returns:
            torch.Tensor: Dice loss
        """
        target = target.unsqueeze(1)
        pred = F.softmax(pred, dim=1)  

        target_one_hot = torch.zeros_like(pred)  
        target_one_hot.scatter_(1, target.long(), 1)  
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))  
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))  
        
        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)  
        dice_loss = 1 - dice_score.mean()  

        return dice_loss
