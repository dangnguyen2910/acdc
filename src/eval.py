import os
import numpy as np 
import matplotlib.pyplot as plt
import torch 
from torch.utils.data import DataLoader

from src.model.unet import UNet3D
from src.dataset import ACDC, ACDCProcessed, JustToTest
from src.metrics import calculate_multiclass_dice

if __name__ == "__main__": 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    device = "cuda"
    model_path = "model/unet3d_6.pth"
    print(f"Using {model_path}")

    model = UNet3D().to(device)

    model.load_state_dict(torch.load(model_path))

    test_dataset = JustToTest("just_to_test/testing", is_testset=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dices = []
    class1_dice = []
    class2_dice = []
    class3_dice = []

    with torch.no_grad(): 
        for i,data in enumerate(test_dataloader): 
            img, gt = data
            img = img.to(device)
            gt = gt.to(device)

            output = model(img)
            output = (output > 0.5).to(torch.long)

            dice, dice_list = calculate_multiclass_dice(
                output.squeeze(), 
                gt.squeeze()
            )
            
            class1_dice.append(dice_list[0])
            class2_dice.append(dice_list[1])
            class3_dice.append(dice_list[2])
            dices.append(dice)  
            
            print(f"{dice_list} Dice: {dice:.3f}")
                      

    mean_dice_class1 = np.mean(class1_dice)
    mean_dice_class2 = np.mean(class2_dice)
    mean_dice_class3 = np.mean(class3_dice)

    print("Mean Dice of class 1: ", mean_dice_class1)
    print("Mean Dice of class 2: ", mean_dice_class2)
    print("Mean Dice of class 3: ", mean_dice_class3)
    print("Mean Dice: ", np.mean(dices))

