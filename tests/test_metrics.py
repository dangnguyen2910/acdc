import pytest
import torch
import numpy as np 
from src.model.model import UNet2D
from src.metrics import calculate_binary_dice, calculate_multiclass_dice

def test_calculate_binary_dice():
    gt = torch.tensor([
        [1,0,0], 
        [1,1,1], 
        [0,1,0]
    ])
    
    pred = torch.tensor([
        [0,0,0],
        [1,0,1], 
        [1,1,1]
    ])
    assert np.round(calculate_binary_dice(gt, pred), 2) == 0.6
    

def test_calculate_binary_dice_empty(): 
    gt = torch.zeros((3,3))
    
    pred = torch.zeros((3,3))
    assert calculate_binary_dice(gt, pred) == 1
    

def test_calculate_multiclass_dice(): 
    gt = torch.ones(2, 2, 3, 3)
    pred = torch.zeros(2,2,3,3)
    dice, dice_list = calculate_multiclass_dice(pred,gt, 2)
    assert dice == (1e-6/(18+1e-6))
    

def test_calculate_multiclass_dice_empty(): 
    gt = torch.zeros(2, 2, 3, 3)
    pred = torch.zeros(2,2,3,3)
    dice, dice_list = calculate_multiclass_dice(pred, gt, 2)
    assert dice == 1
