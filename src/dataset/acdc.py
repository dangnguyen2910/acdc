
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
sitk.ProcessObject.SetGlobalWarningDisplay(False)
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
        self.df = self.__make_dataframe() 
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

        # For retaining the depth size when resize
        num_layers = img.size(1)
        transform = tio.Compose([
            tio.Resize((num_layers, 352,352)), 
            tio.CropOrPad((10, 224, 224)),
            # tio.Crop((0,0,64,64,64,64)), 
            tio.ZNormalization()
        ])
        
        gt_transform = tio.Compose([
            tio.Resize((num_layers, 352,352)), 
            tio.CropOrPad((10, 224, 224)),
            # tio.Crop((0,0,64,64,64,64))
        ])
        
        img = transform(img)
        gt = gt_transform(gt).to(torch.long)
        
        # No augment if this is testset
        if (self.is_testset): 
            return img, gt
        
        # Augmentation if this is training set
        subject = tio.Subject(
            image = tio.ScalarImage(tensor = img), 
            mask = tio.LabelMap(tensor = gt)
        )

        augment = tio.Compose([
            tio.RandomFlip(axes=(1,2)), 
            tio.RandomElasticDeformation(max_displacement=1), 
            # tio.RandomMotion(translation=2, num_transforms=1), 
            tio.RandomGamma()
        ])
        subject = augment(subject)
        return subject.image.tensor, subject.mask.tensor


    def __len__(self): 
        return self.df.shape[0]


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