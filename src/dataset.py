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

from src.model.model import UNet3D

warnings.filterwarnings("ignore")

class ACDC(Dataset): 
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.patient_folders = [
            os.path.join(data_path, folder) for folder in os.listdir(data_path)
        ]
        self.df = self.make_dataframe()
        self.transform = tio.Compose([
            tio.RescaleIntensity(out_min_max=(0,1)), 
            tio.Resize((10,320,320))
        ])

        self.gt_transform = tio.Compose([
            tio.Resize((10,320,320))
        ])


    def __getitem__(self, index):
        img_path = self.df.iloc[index, 0]
        gt_path = self.df.iloc[index, 1]
        
        # img = self.load_nifti_image(img_path)
        # gt = self.load_nifti_image(gt_path)

        img = self.resample_3d_image(img_path)
        gt = self.resample_3d_image(gt_path)

        img = sitk.GetArrayFromImage(img)
        gt = sitk.GetArrayFromImage(gt)

        img = torch.tensor(img).unsqueeze(0)
        img = self.transform(img)
        
        gt = torch.tensor(gt).unsqueeze(0)
        gt = self.gt_transform(gt).to(torch.long)
        
        return img, gt

    # def __getitem__(self, index): 
        '''
        An alternative version if you want to separate ed and es
        '''
         # patient_folder_path = self.patient_folders[index]
        # files = [os.path.join(patient_folder_path, file) for file in os.listdir(patient_folder_path)]

        # # ED, ES, gt_ED, gt_ES
        # ed_path = next((file for file in files if "frame01" in file and "gt" not in file), None)
        # es_path = next((file for file in files if "frame01" not in file and "frame" in file and "gt" not in file), None)

        # ed_gt_path = next((file for file in files if "frame01" in file and "gt" in file), None)
        # es_gt_path = next((file for file in files if "frame01" not in file and "frame" in file and "gt" in file), None)

        # ed = self.load_nifti_image(ed_path)
        # es = self.load_nifti_image(es_path)
        # ed_gt = self.load_nifti_image(ed_gt_path)
        # es_gt = self.load_nifti_image(es_gt_path)

        # return (ed, ed_gt, es, es_gt)

    def __len__(self): 
        return self.df.shape[0]


    def load_nifti_image(self, file_path):
        img = nib.load(file_path)
        data = img.get_fdata()
        return data

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
        })
        return df


    def extract_code(self, path): 
        info = pd.read_csv(path, header = None, delimiter=":")
        ed_frameId = (int(info.iloc[0,1]))
        es_frame_Id = (int(info.iloc[1,1]))

        return ed_frameId, es_frame_Id


    def resample_3d_image(self, image_path, new_spacing=(1.25, 1.25, 10.0), interpolator=sitk.sitkLinear):
        # Load image
        image = sitk.ReadImage(image_path)

        # Get original spacing and size
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        # Compute new size to preserve field of view
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / new_spacing[i])))
            for i in range(3)
        ]

        # Define resampling filter
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())

        # Apply resampling
        resampled_image = resampler.Execute(image)
        
        return resampled_image

        
class ACDCProcessed(ACDC): 
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
        
        df = pd.DataFrame({
            "img": img_list, 
            "gt": gt_list
        })
        return df

# For testing only
if __name__ == "__main__": 
    dataset = ACDCProcessed("processed/training")
    
    print("Data size:", len(dataset))
    for i in range(len(dataset)):
        print("New image")
        img, gt = dataset[i]

        img = img.unsqueeze(0)
        gt = gt.unsqueeze(0).squeeze(1)
        

        print("Image shape:", img.size())
        print("GT shape:", gt.size())

        # for j in range(img.shape[2]):
        #     print(f"Layer {j}")
        #     img_tmp = img.squeeze().permute(1,2,0).cpu()[:,:,j]
        #     # output = output.squeeze().permute(1,2,0).detach().numpy()[:,:,0]
        #     gt_tmp = gt.squeeze().permute(1,2,0).cpu()[:,:,j]

        #     fig, ax = plt.subplots(1,2)
        #     ax[0].imshow(img_tmp, 'gray')
        #     ax[1].imshow(gt_tmp)
        #     plt.show()
        #     plt.close()
