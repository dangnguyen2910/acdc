import pytest 
import torch

from src.dataset.acdc import ACDC
from src.dataset.acdc_processed import ACDCProcessed
from src.dataset.just_to_test import JustToTest


@pytest.fixture
def dataset(): 
    dataset = ACDC("database/training", is_testset=False)
    return dataset 

def test_dataset_make_dataframe(dataset): 
    df = dataset.df 
    assert df.iloc[0,0] == "database/training/patient001/patient001_frame01.nii.gz"
    assert df.iloc[2,0] == "database/training/patient002/patient002_frame01.nii.gz"
    assert df.shape == (160,2)
    
def test_dataset_shape(dataset): 
    img, gt = dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])


@pytest.fixture
def processed_dataset(): 
    processed_dataset = ACDCProcessed("processed/training", is_testset=False)
    return processed_dataset 

def test_processed_dataset_make_dataframe(processed_dataset): 
    df = processed_dataset.df 
    assert df.iloc[0,0] == "processed/training/patient001/patient001_ED_processed.nii.gz"
    assert df.iloc[6,0] == "processed/training/patient002/patient002_ED_processed.nii.gz"
    assert df.shape == (480,2)
    
def test_processed_dataset_shape(processed_dataset): 
    img, gt = processed_dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])

    
@pytest.fixture
def processed_dataset(): 
    processed_dataset = ACDCProcessed("processed/training", is_testset=False)
    return processed_dataset 

def test_processed_dataset_make_dataframe(processed_dataset): 
    df = processed_dataset.df 
    assert df.iloc[0,0] == "processed/training/patient001/patient001_ED_processed.nii.gz"
    assert df.iloc[6,0] == "processed/training/patient002/patient002_ED_processed.nii.gz"
    assert df.shape == (480,2)
    
def test_processed_dataset_shape(processed_dataset): 
    img, gt = processed_dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])
    

@pytest.fixture
def just_to_test_dataset(): 
    just_to_test_dataset = JustToTest("just_to_test/training", is_testset=False)
    return just_to_test_dataset 

def test_just_to_test_dataset_make_dataframe(just_to_test_dataset): 
    df = just_to_test_dataset.df 
    assert df.iloc[0,0] == "just_to_test/training/patient001/patient001_ED_processed.nii.gz"
    assert df.iloc[4,0] == "just_to_test/training/patient002/patient002_ED_processed.nii.gz"
    assert df.shape == (320,2)
    
def test_just_to_test_dataset_shape(just_to_test_dataset): 
    img, gt = just_to_test_dataset[0]
    assert img.size() == torch.Size([1, 10, 224, 224])
    assert gt.size() == torch.Size([3, 10, 224, 224])