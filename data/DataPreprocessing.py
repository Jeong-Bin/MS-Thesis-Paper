import os
import glob
import pandas as pd
import cv2
from torch.utils.data import Dataset
from sklearn import preprocessing
import learn2learn as l2l
from learn2learn.data.transforms import NWays, KShots, LoadData, RemapLabels


# class DataPreprocessing(torch.utils.data.TensorDataset):
#     def __init__(self, dataset):
#         data = []
#         labels = []
#         for x, y in dataset:
#             data.append(x.unsqueeze(0))
#             labels.append(y)
#         data = torch.cat(data, dim=0)
#         labels = torch.tensor(labels)

#         super(DataPreprocessing, self).__init__(data, labels)
#         if hasattr(dataset, '_bookkeeping_path'):
#             self._bookkeeping_path = dataset._bookkeeping_path

#     def __getitem__(self, index):
#         x, y = super(DataPreprocessing, self).__getitem__(index)
#         return x, y

def Meta_Transforms(dataset, way, shot, query, num_tasks):
    meta_dataset = l2l.data.MetaDataset(dataset)

    FewShot_transforms = [
        NWays(meta_dataset, way),
        KShots(meta_dataset, shot + query),
        LoadData(meta_dataset),
        RemapLabels(meta_dataset)]

    tasksets = l2l.data.TaskDataset(dataset = meta_dataset, 
                                    task_transforms = FewShot_transforms,
                                    num_tasks = num_tasks)

    return tasksets


def make_df(root, mode):
    path = os.path.join(root, mode)
    all_img_list = glob.glob(f'{path}/*/*')
    df = pd.DataFrame(columns=['img_path', 'label'])
    df['img_path'] = all_img_list
    df['label'] = df['img_path'].apply(lambda x : str(x).split('/')[-2])

    # label encoding
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    return df


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)
    

