import pytest 
import torch

from src.dataset import ACDCProcessed

@pytest.fixture()
def processed_dataset(): 
    processed_dataset = ACDCProcessed("processed/training", is_testset=False)
    return processed_dataset 

def test_make_dataframe(processed_dataset): 
    df = processed_dataset.df 
    assert df.iloc[0,0] == "processed/training/patient001/patient001_ED_processed.nii.gz"
    assert df.iloc[6,0] == "processed/training/patient002/patient002_ED_processed.nii.gz"
    assert df.shape == (480,2)
    
def test_processed_dataset_shape(processed_dataset): 
    img, gt = processed_dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])
    
    
@pytest.fixture()
def new_processed_dataset(): 
    new_processed_dataset = ACDCProcessed("just_to_test/training", is_testset=False)
    return new_processed_dataset 

def test_make_dataframe(new_processed_dataset): 
    df = new_processed_dataset.df 
    assert df.iloc[0,0] == "just_to_test/training/patient001/patient001_ED_processed.nii.gz"
    assert df.iloc[6,0] == "just_to_test/training/patient002/patient002_ED_processed.nii.gz"
    assert df.shape == (480,2)
    
def test_new_processed_dataset_shape(new_processed_dataset): 
    img, gt = new_processed_dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])