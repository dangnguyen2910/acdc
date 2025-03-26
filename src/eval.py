import os
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader

from src.model.model import UNet3D
from src.dataset import ACDC, ACDCProcessed


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
    
    intersection = (pred_binary * gt_binary).sum()
    sum_cardinality = pred_binary.sum() + gt_binary.sum()
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
        
    return dice_list.mean(), dice_list

if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    device = "cuda"
    model_path = "model/unet3d_2"
    print(f"Using {model_path}")

    model = UNet3D(in_channels=1, out_channels=4, is_segmentation=False).to(device)


    state_dict = torch.load("model/unet3d_2.pth")

    model.load_state_dict(state_dict)

    test_dataset = ACDCProcessed("processed/testing", is_testset=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dices = []
    class0_dice = []
    class1_dice = []
    class2_dice = []
    class3_dice = []

    with torch.no_grad(): 
        for i,data in enumerate(test_dataloader): 
            img, gt = data
            img = img.to(device)
            gt = gt.to(device)

            print("Inferencing")
            output = model(img)
            output = torch.argmax(torch.softmax(output, dim=1), dim=1, keepdim=True)

            # print(np.unique(output.cpu().numpy()))
            # for j in range(10): 
            #     fig, ax = plt.subplots(1,3)
            #     ax[0].imshow(img.squeeze().permute(1,2,0).cpu().numpy()[:,:,j], 'gray')
            #     ax[1].imshow(gt.squeeze().permute(1,2,0).cpu().numpy()[:,:,j])
            #     ax[2].imshow(output.squeeze().permute(1,2,0).cpu().numpy()[:,:,j])
            #     plt.show()
            #     plt.close()

            dice_scores = dice_coefficient(output, gt, 4)
            class0, class1, class2, class3 = dice_scores
            print(class0.item(), class1.item(), class2.item(), class3.item())
            class0_dice.append(class0.cpu())
            class1_dice.append(class1.cpu())
            class2_dice.append(class2.cpu())
            class3_dice.append(class3.cpu())

    # mean_dice_class0 = np.mean(class0_dice)
    mean_dice_class1 = np.mean(class1_dice)
    mean_dice_class2 = np.mean(class2_dice)
    mean_dice_class3 = np.mean(class3_dice)

    # print("Mean Dice of class 0", mean_dice_class0)
    print("Mean Dice of class 1", mean_dice_class1)
    print("Mean Dice of class 2", mean_dice_class2)
    print("Mean Dice of class 3", mean_dice_class3)





def dice_coefficient(pred, target, num_classes, epsilon=1e-6):
    """ 
    Calculate the dice coefficient
    
    Parameter: 
        pred: binary multichannels predicted mask (C, L, H, W)
        target: binary multichannel ground truth mask (C, L, H, W)
        num_class: number of classes
        epsilon = 10^-6: smoothing factor
    Output: 
        float: 
    """
    dice_scores = []
    
    
    # dice_scores = []
    
    # for class_idx in range(num_classes):
    #     pred_binary = (pred == class_idx)  
    #     target_binary = (target == class_idx) 
        
    #     intersection = (pred_binary & target_binary).sum().float()
    #     union = pred_binary.sum().float() + target_binary.sum().float()
        
    #     dice = (2.0 * intersection + epsilon) / (union + epsilon)
    #     dice_scores.append(dice)
    
    # return dice_scores
