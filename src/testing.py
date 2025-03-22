import torchio as tio
import nibabel as nib
import SimpleITK as sitk
import pandas as pd 
import os


def extract_code(path): 
    info = pd.read_csv(path, header = None, delimiter=":")
    ed_frameId = (int(info.iloc[0,1]))
    es_frame_Id = (int(info.iloc[1,1]))

    return ed_frameId, es_frame_Id

def make_dataframe(): 
    '''
    Create a dataframe with 2 cols: image path and gt_path
    '''
    img_list = []
    gt_list = []
            
    for root, _, files in os.walk("database/training"): 
        files = [os.path.join(root, file) for file in files]

        if (len(files) == 0): 
            continue

        patient_id = os.path.basename(root)
        ed_frameId, es_frameId = extract_code(os.path.join(root, "Info.cfg"))

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

def load_nifti_image(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data

def resample_3d_image(image_path, new_spacing=(1.25, 1.25, 10.0), interpolator=sitk.sitkLinear):
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

df = make_dataframe()
img = resample_3d_image(df.iloc[0,1])
print(img.GetSpacing())
print(sitk.GetArrayFromImage(img).shape)
# gt = load_nifti_image(df[0,1])

