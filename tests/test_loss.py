import pytest
import torch
import numpy as np 

from src.loss import DiceLoss3D

@pytest.fixture()
def loss_fn(): 
    loss_fn = DiceLoss3D()
    return loss_fn

def test_loss_forward_fit_all(loss_fn): 
    pred = torch.ones(1,2,2,3,3)
    gt = torch.ones(1,2,2,3,3)
    loss = loss_fn(pred, gt)
    assert loss.item() == 0
    
def test_loss_forward_fail_all(loss_fn): 
    pred = torch.ones(1,2,2,3,3)
    gt = torch.zeros(1,2,2,3,3)
    loss = loss_fn(pred, gt)
    assert loss.item() == 1
    