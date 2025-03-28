import pytest 
import torch

from src.model.unet import *

def test_encoder(): 
    x = torch.randn((1,1,10,20,20))
    encoder = Encoder(1, 24)
    output = encoder(x)
    assert output.shape == (1,24,10, 20,20)
    

def test_decoder(): 
    x = torch.randn((1,10,10,6,6))
    y = torch.randn((1, 5, 10, 12, 12))
    decoder = Decoder(10, 5)
    output = decoder(x, y)
    assert output.shape == (1, 5, 10 ,12, 12)
    
def test_analysis(): 
    x = torch.randn((1,1,10,64, 64))
    model = Analysis()
    y = model(x)
    assert y[0].shape == (1, 26, 10, 64, 64)
    assert y[1].shape == (1, 52, 10, 32, 32)
    assert y[2].shape == (1, 104, 10, 16, 16)
    assert y[3].shape == (1, 208, 10, 8, 8) 
    assert y[4].shape == (1, 416, 10, 4, 4)
    

def test_synthesis(): 
    x = torch.randn((1,1,10,64, 64))
    y = Analysis()(x)
    model = Synthesis()
    y = model(y[4], y)
    assert y.shape == (1, 26, 10, 64, 64)
    
    
def test_unet3d(): 
    x = torch.randn((1,1,10,64, 64))
    model = UNet3D()
    y = model(x)
    assert y.shape == (1, 3, 10, 64, 64)