import torchio as tio
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

df = make_dataframe()

subjects = [
    tio.Subject(
        image = tio.ScalarImage(df.iloc[i,0]), 
        gt = tio.LabelMap(df.iloc[i,1])
    )
    for i in range(df.shape[0])
]

print(len(subjects))
dataset = tio.SubjectsDataset(subjects)
loader = tio.SubjectsLoader(
    dataset, 
    batch_size = 1, 
    shuffle = True
)

for data in loader: 
    print(type(data))
    inputs = data['image'][tio.DATA]
    gt = data['gt'][tio.DATA]

    print(inputs.size())