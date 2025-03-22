import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader

from src.model.model import UNet3D
from src.dataset import ACDC

def dice_coefficient(pred, target, num_classes, epsilon=1e-6):
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_binary = (pred == class_idx)  
        target_binary = (target == class_idx) 
        
        intersection = (pred_binary & target_binary).sum().float()
        union = pred_binary.sum().float() + target_binary.sum().float()
        
        dice = (2.0 * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice)
    
    return dice_scores

device = "cpu"
model = UNet3D(in_channels=1, out_channels=4, is_segmentation=False).to(device)


state_dict = torch.load("model/unet3d.pth")

model.load_state_dict(state_dict)

test_dataset = ACDC("database/testing")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

dices = []
class0_dice = []
class1_dice = []
class2_dice = []
class3_dice = []

with torch.no_grad(): 
    for i,data in enumerate(test_dataloader): 
        if (i == 10): 
            break

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
        print(class0, class1, class2, class3)
        class0_dice.append(class0)
        class1_dice.append(class1)
        class2_dice.append(class2)
        class3_dice.append(class3)

# mean_dice_class0 = np.mean(class0_dice)
mean_dice_class1 = np.mean(class1_dice)
mean_dice_class2 = np.mean(class2_dice)
mean_dice_class3 = np.mean(class3_dice)

# print("Mean Dice of class 0", mean_dice_class0)
print("Mean Dice of class 1", mean_dice_class1)
print("Mean Dice of class 2", mean_dice_class2)
print("Mean Dice of class 3", mean_dice_class3)
