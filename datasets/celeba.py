import os
import torch
import numpy as np
import h5py
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import random

from datasets.base import BaseLoader


class CelebaRaw(Dataset):
    def __init__(self, image_list, labels_dict, h5py_file_path, target_idx):
        self.image_list = image_list
        self.labels_dict = labels_dict
        self.h5py_file_path = h5py_file_path 
        self.target_idx = target_idx
        self.h5py_file = None

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.h5py_file is None:
            self.h5py_file = h5py.File(self.h5py_file_path, 'r', swmr=True)

        image_name = self.image_list[idx]
        img = np.array(self.h5py_file[image_name][()])
        img = torch.from_numpy(img).permute(2, 0, 1).float() 

        label_info = self.labels_dict[image_name]
        label = torch.tensor(label_info[self.target_idx], dtype=torch.long)
        sens = torch.tensor(label_info[39], dtype=torch.long)  # Male

        return img, label, sens
    
    def __del__(self):
        if self.h5py_file is not None:
            self.h5py_file.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5py_file'] = None 
        return state


class CelebALoader(BaseLoader):
    def load(self, batch_size=256, random_state=None):
        h5py_path = os.path.join(self.data_root, "celeba/celeba_preprocessed.h5")

        prefetch_factor = 2

        with open(os.path.join(self.data_root, "celeba/labels_dict"), 'rb') as f:
            attributes_dict = pickle.load(f)

        all_list = list(attributes_dict.keys())

        def split_by_sens(image_list, labels_dict):
            sens_0_list = []
            sens_1_list = []

            for image_name in image_list:
                sens = labels_dict[image_name][39]  # Sensitive attribute (Male)
                if sens == 0:
                    sens_0_list.append(image_name)
                else:
                    sens_1_list.append(image_name)

            return sens_0_list, sens_1_list

        sens_0_list, sens_1_list = split_by_sens(all_list, attributes_dict)
    
        train_list0, test_list0 = train_test_split(sens_0_list, test_size=0.1, random_state=random_state)
        train_list0, val_list0 = train_test_split(train_list0, test_size=0.1, random_state=random_state)
        train_list0, thresh_list0 = train_test_split(train_list0, test_size=0.2, random_state=random_state)
        train_list1, test_list1 = train_test_split(sens_1_list, test_size=0.1, random_state=random_state)
        train_list1, val_list1 = train_test_split(train_list1, test_size=0.1, random_state=random_state)
        train_list1, thresh_list1 = train_test_split(train_list1, test_size=0.2, random_state=random_state)

        train_dataset0 = CelebaRaw(train_list0, attributes_dict, h5py_path, target_idx=2)  # Target: attractive
        val_dataset0 = CelebaRaw(val_list0, attributes_dict, h5py_path, target_idx=2)
        test_dataset0 = CelebaRaw(test_list0, attributes_dict, h5py_path, target_idx=2)
        thresh_dataset0 = CelebaRaw(thresh_list0, attributes_dict, h5py_path, target_idx=2)
        train_dataset1 = CelebaRaw(train_list1, attributes_dict, h5py_path, target_idx=2)
        val_dataset1 = CelebaRaw(val_list1, attributes_dict, h5py_path, target_idx=2)
        test_dataset1 = CelebaRaw(test_list1, attributes_dict, h5py_path, target_idx=2)
        thresh_dataset1 = CelebaRaw(thresh_list1, attributes_dict, h5py_path, target_idx=2)

        train_loader0 = DataLoader(dataset=train_dataset0, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        val_loader0 = DataLoader(dataset=val_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        test_loader0 = DataLoader(dataset=test_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        thresh_loader0 = DataLoader(dataset=thresh_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

        train_loader1 = DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        val_loader1 = DataLoader(dataset=val_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        test_loader1 = DataLoader(dataset=test_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        thresh_loader1 = DataLoader(dataset=thresh_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

        return train_loader0, val_loader0, test_loader0, thresh_loader0, train_loader1, val_loader1, test_loader1, thresh_loader1
