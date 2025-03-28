
import numpy as np 
import pandas as pd
import warnings
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchio as tio
import torch
import nibabel as nib
import SimpleITK as sitk
import os

from src.dataset.acdc import ACDC

warnings.filterwarnings("ignore")

class ACDCProcessed(ACDC): 
    """ 
    Subclass of ACDC to deal with processed dataset
    
    Parameter: 
        data_path (string): path of training or testing
        is_testset (bool): True if loading validation or test set    
    
    """
    
    def __init__(self, data_path, is_testset): 
        self.data_path = data_path
        self.is_testset = is_testset
        self.df = self.__make_dataframe()

    
    def __make_dataframe(self): 
        '''
        Create a dataframe with 2 cols: image path and gt_path
        '''
        img_list = []
        gt_list = []
                
        for root, _, files in os.walk(self.data_path): 
            files = [os.path.join(root, file) for file in files]

            if (len(files) == 0): 
                continue

            patient_id = os.path.basename(root)

            ed_path = os.path.join(root, patient_id + "_ED"  + "_processed.nii.gz")
            ed_gt_path = os.path.join(root, patient_id + "_ED_gt" + "_processed.nii.gz")

            es_path = os.path.join(root, patient_id + "_ES" + "_processed.nii.gz")
            es_gt_path = os.path.join(root, patient_id + "_ES_gt" + "_processed.nii.gz")

            img_list.append(ed_path)
            img_list.append(es_path)
            gt_list.append(ed_gt_path)
            gt_list.append(es_gt_path)

            if self.is_testset == False: 
                ed_path_a0 = os.path.join(root, patient_id + "_ED"  + "_processed_augmented0.nii.gz")
                ed_path_a1 = os.path.join(root, patient_id + "_ED"  + "_processed_augmented1.nii.gz")

                ed_gt_path_a0 = os.path.join(root, patient_id + "_ED_gt" + "_processed_augmented0.nii.gz")
                ed_gt_path_a1 = os.path.join(root, patient_id + "_ED_gt" + "_processed_augmented1.nii.gz")

                es_path_a0 = os.path.join(root, patient_id + "_ES" + "_processed_augmented0.nii.gz")
                es_path_a1 = os.path.join(root, patient_id + "_ES" + "_processed_augmented1.nii.gz")

                es_gt_path_a0 = os.path.join(root, patient_id + "_ES_gt" + "_processed_augmented0.nii.gz")
                es_gt_path_a1 = os.path.join(root, patient_id + "_ES_gt" + "_processed_augmented1.nii.gz")
            
                img_list.append(ed_path_a0)
                img_list.append(ed_path_a1)

                img_list.append(es_path_a0)
                img_list.append(es_path_a1)

                gt_list.append(ed_gt_path_a0)
                gt_list.append(ed_gt_path_a1)

                gt_list.append(es_gt_path_a0)
                gt_list.append(es_gt_path_a1)
        
        df = pd.DataFrame({
            "img": img_list, 
            "gt": gt_list
        }).sort_values(by="img")
        return df