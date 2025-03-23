import torch
import torch.nn as nn

class DiceLoss3D(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss3D, self).__init__()
        self.smooth = smooth  

    def forward(self, preds, targets):
        """
        Compute Dice Loss for 3D images.
        
        Args:
        preds (torch.Tensor): Logits or probabilities of shape (N, C, D, H, W).
        targets (torch.Tensor): Ground truth masks of shape (N, C, D, H, W), with values in {0,1}.
        
        Returns:
        torch.Tensor: Dice loss value.
        """
        preds = torch.sigmoid(preds)


        preds = preds.view(preds.shape[0], preds.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)

        intersection = (preds * targets).sum(dim=2) 
        union = preds.sum(dim=2) + targets.sum(dim=2) 

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)  
        dice_loss = 1 - dice_score.mean()  

        return dice_loss
