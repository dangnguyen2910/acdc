import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss3D(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss3D, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, gt):
        """
        Compute Dice Loss for multi-class 3D segmentation.  
        The dimension of pred and gt should be equal 

        Args:
            pred (torch.Tensor): Predicted sigmoid output (N, C, L, H, W)
            gt (torch.Tensor): Ground truth class indices (N, C, L, H, W)

        Returns:
            torch.Tensor: Dice loss
        """
        # print(pred.size())
        # print(gt.size())
        # pred_flat = pred.contiguous().view(-1)
        # gt_flat = pred.contiguous().view(-1)
        # print(pred_flat.size())
        
        intersection = (pred * gt).sum(dim = (2,3,4))

        pred_sum = (pred * pred).sum(dim = (2,3,4))
        gt_sum = (gt * gt).sum(dim = (2,3,4))

        numerator = 2 * intersection + self.epsilon
        denom = pred_sum + gt_sum + self.epsilon
        
        dice = numerator/denom
        return 1 - dice.mean()
        
        # target = target.unsqueeze(1)
        # pred = F.softmax(pred, dim=1)  

        # target_one_hot = torch.zeros_like(pred)  
        # target_one_hot.scatter_(1, target.long(), 1)  
        
        # intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))  
        # union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))  
        
        # dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)  
        # dice_loss = 1 - dice_score.mean()  

        # return dice_loss
