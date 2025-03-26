import numpy as np 

def calculate_binary_dice(pred_binary, gt_binary):
    """ 
    Compute dice coefficient for two binary masks.
    
    Parameters:
    -----------
    pred_binary (torch.tensor): Binary mask for predicted panel. (1, L, H, W)
    gt_binary (torch.tensor): Binary mask for ground truth panel. (1, L, H, W)
    
    Returns:
    --------
    float: Dice coef value.
    """
    epsilon = 1e-6
    
    intersection = (pred_binary * gt_binary).sum().item()
    sum_cardinality = (pred_binary.sum() + gt_binary.sum()).item()
    return ((2*intersection + epsilon) / (sum_cardinality + epsilon)) 


def calculate_multiclass_dice(pred, gt, num_class = 3): 
    """ 
    Compute dice coefficient for two multichannel binary mask.
    
    Parameters:
    -----------
    pred (torch.tensor): Predicted mask (num_class, L, H, W)
    gt (torch.tensor): Ground truth mask (num_class, L, H, W)
    num_class (int): Number of classes, default = 3
    
    Returns:
    --------
    list: List of dice values for each class (C)
    float: Dice coef value.
    """
    dice_list = []
    
    for c in range(num_class): 
        pred_binary = pred[c, :, :, :]
        gt_binary = gt[c, :, :, :]
        dice = calculate_binary_dice(pred_binary, gt_binary)
        dice_list.append(dice)
        
    return np.mean(dice_list), dice_list