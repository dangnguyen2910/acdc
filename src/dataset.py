import numpy as np 
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
import torchio as tio
import torch
import nibabel as nib
import SimpleITK as sitk
import os

warnings.filterwarnings("ignore")

class ACDC(Dataset): 
    """ 
    Class for ACDC original dataset
    
    Parameter: 
        data_path (string): path to training or testing set
    """
    def __init__(self, data_path, is_testset):
        super().__init__()
        self.data_path = data_path
        self.is_testset = is_testset
        self.patient_folders = [
            os.path.join(data_path, folder) for folder in os.listdir(data_path)
        ]
        self.df = self.make_dataframe() 
        self.num_layers = 0


    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        gt_path = self.df.iloc[index, 1]

        obj_ids = torch.tensor([1,2,3]).to(torch.long)
        
        img = sitk.ReadImage(img_path)
        gt = sitk.ReadImage(gt_path)
        
        img = sitk.GetArrayFromImage(img)
        gt = sitk.GetArrayFromImage(gt)

        img = torch.tensor(img).unsqueeze(0)
        gt = torch.tensor(gt).unsqueeze(0)
        gt = (gt == obj_ids[:,None,None,None])
        
        num_layers = img.size(1)
        
        transform = tio.Compose([
            tio.Resampling((1.25, 1.25, 10)), 
            tio.Resize((num_layers, 352,352)), 
            # tio.CropOrPad((10, 352, 352)),
            tio.Crop((0,0,64,64,64,64)), 
            tio.ZNormalization()
        ])
        
        gt_transform = tio.Compose([
            tio.Resample((1.25, 1.25, 10)), 
            tio.Resize((num_layers, 352,352)), 
            # tio.CropOrPad((10, 352, 352)),
            tio.Crop((0,0,64,64,64,64))
        ])
        
        augment = tio.Compose([
            tio.RandomFlip(axes=(1,2)), 
            tio.RandomElasticDeformation(max_displacement=1), 
            tio.RandomMotion(translation=2, num_transforms=1), 
            tio.RandomGamma()
        ])
        
        img = transform(img)
        gt = gt_transform(gt).to(torch.long)
        
        subject = tio.Subject(
            image = tio.ScalarImage(tensor = img), 
            mask = tio.LabelMap(tensor = gt)
        )
        
        if (not self.is_testset): 
            subject = augment(subject)
        
        return subject.image.tensor, subject.mask.tensor


    def __len__(self): 
        return self.df.shape[0]


    def make_dataframe(self): 
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
            ed_frameId, es_frameId = self.extract_code(os.path.join(root, "Info.cfg"))

            ed_path = os.path.join(root, patient_id + "_frame0" + str(ed_frameId) + ".nii.gz")
            ed_gt_path = os.path.join(root, patient_id + "_frame0" + str(ed_frameId) + "_gt.nii.gz")

            if (es_frameId > 9): 
                es_path = os.path.join(root, patient_id + "_frame" + str(es_frameId) + ".nii.gz")
                es_gt_path = os.path.join(root, patient_id + "_frame" + str(es_frameId) + "_gt.nii.gz")
            else: 
                es_path = os.path.join(root, patient_id + "_frame0" + str(es_frameId) + ".nii.gz")
                es_gt_path = os.path.join(root, patient_id + "_frame0" + str(es_frameId) + "_gt.nii.gz")
            

            img_list.append(ed_path)
            img_list.append(es_path)
            gt_list.append(ed_gt_path)
            gt_list.append(es_gt_path)
        
        df = pd.DataFrame({
            "img": img_list, 
            "gt": gt_list
        }).sort_values(by="img")
        return df


    def extract_code(self, path): 
        info = pd.read_csv(path, header = None, delimiter=":")
        ed_frameId = (int(info.iloc[0,1]))
        es_frame_Id = (int(info.iloc[1,1]))

        return ed_frameId, es_frame_Id

        
class ACDCProcessed(ACDC): 
    """ 
    Subclass of ACDC to deal with processed dataset
    
    Parameter: 
        data_path (string): path of training or testing
        is_testset (bool): True if loading validation or test set    
    
    """
    
    def __init__(self, data_path, is_testset): 
        super().__init__(data_path, is_testset)

    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        gt_path = self.df.iloc[index, 1]

        obj_ids = torch.tensor([1,2,3]).to(torch.long)
        
        img = sitk.ReadImage(img_path)
        gt = sitk.ReadImage(gt_path)
        
        img = sitk.GetArrayFromImage(img)
        gt = sitk.GetArrayFromImage(gt)

        img = torch.tensor(img).unsqueeze(0)
        gt = torch.tensor(gt).unsqueeze(0)
        gt = (gt == obj_ids[:,None,None,None])
        
        num_layers = img.size(1)
        
        transform = tio.Compose([
            tio.ZNormalization(),
            tio.Resize((num_layers, 352,352)), 
            # tio.CropOrPad((10, 352, 352)),
            tio.Crop((0,0,64,64,64,64))
        ])
        
        gt_transform = tio.Compose([
            tio.Resize((num_layers, 352,352)), 
            # tio.CropOrPad((10, 352, 352)),
            tio.Crop((0,0,64,64,64,64))
        ])
        
        augment = tio.Compose([
            tio.RandomFlip(axes=(1,2)), 
            tio.RandomElasticDeformation(max_displacement=1), 
            tio.RandomMotion(translation=2, num_transforms=1), 
            tio.RandomGamma()
        ])
        
        img = transform(img)
        gt = gt_transform(gt).to(torch.long)
        
        subject = tio.Subject(
            image = tio.ScalarImage(tensor = img), 
            mask = tio.LabelMap(tensor = gt)
        )
        
        # if (not self.is_testset): 
        #     subject = augment(subject)
        
        return subject.image.tensor, subject.mask.tensor


    def make_dataframe(self): 
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
    
    
class JustToTest(ACDCProcessed): 
    """ 
    Subclass of ACDC to deal with processed dataset
    
    Parameter: 
        data_path (string): path of training or testing
        is_testset (bool): True if loading validation or test set    
    
    """
    def __init__(self, data_path, is_testset): 
        super().__init__(data_path, is_testset)
    
    def make_dataframe(self): 
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
                ed_gt_path_a0 = os.path.join(root, patient_id + "_ED_gt" + "_processed_augmented0.nii.gz")
                es_path_a0 = os.path.join(root, patient_id + "_ES" + "_processed_augmented0.nii.gz")
                es_gt_path_a0 = os.path.join(root, patient_id + "_ES_gt" + "_processed_augmented0.nii.gz")
            
                img_list.append(ed_path_a0)
                img_list.append(es_path_a0)
                gt_list.append(ed_gt_path_a0)
                gt_list.append(es_gt_path_a0)
        
        df = pd.DataFrame({
            "img": img_list, 
            "gt": gt_list
        }).sort_values(by="img")
        return df
