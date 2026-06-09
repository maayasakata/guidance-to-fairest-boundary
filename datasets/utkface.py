import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import PIL
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import random

from datasets.base import BaseLoader

class UTKFaceRaw(Dataset):
    def __init__(self, images, labels, target):
        self.images = images
        self.labels = labels
        self.target = target

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        file_name = self.images[idx]
        attr_info = self.labels.loc[file_name]

        image_name = os.path.join(self.data_root, file_name)
        img = np.array(Image.open(image_name).convert("RGB"))
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # [H, W, C] -> [C, H, W]
        label = attr_info[self.target] 
        sens = attr_info["gender"] 
        label = torch.tensor(label).long()
        sens = torch.tensor(sens).long()

        return img, label, sens
    

class UTKFaceLoader(BaseLoader):
    def load(self, batch_size=256, random_state=None):
        prefetch_factor = 2

        target = "age"
        img_path = os.path.join(self.data_root, "utkface/UTKFace")
        all_img = os.listdir(img_path)

        def split_by_sens(image_list):
            sens_0_img = []
            sens_1_img = []
            sens_0_label = []
            sens_1_label = []
            for image_name in image_list:
                parts = image_name.split("_")
                age = int(parts[0])
                gender = int(parts[1])
                age_label = int(age >= 30)
                if gender == 0: # female
                    sens_0_img.append(image_name)
                    sens_0_label.append({"file": image_name, "age": age_label, "gender": 0})
                else:
                    sens_1_img.append(image_name)
                    sens_1_label.append({"file": image_name, "age": age_label, "gender": 1})

            sens_0_label = pd.DataFrame(sens_0_label).set_index("file")
            sens_1_label = pd.DataFrame(sens_1_label).set_index("file")

            return sens_0_img, sens_0_label, sens_1_img, sens_1_label
        
        sens_0_list, sens_0_label, sens_1_list, sens_1_label = split_by_sens(all_img)
        train_list0, test_list0 = train_test_split(sens_0_list, test_size=0.1, random_state=random_state)
        train_list0, val_list0 = train_test_split(train_list0, test_size=0.1, random_state=random_state)
        train_list0, thresh_list0 = train_test_split(train_list0, test_size=0.2, random_state=random_state)
        train_list1, test_list1 = train_test_split(sens_1_list, test_size=0.1, random_state=random_state)
        train_list1, val_list1 = train_test_split(train_list1, test_size=0.1, random_state=random_state)
        train_list1, thresh_list1 = train_test_split(train_list1, test_size=0.2, random_state=random_state)

        train_dataset0 = UTKFaceRaw(train_list0, sens_0_label, target=target)
        val_dataset0 = UTKFaceRaw(val_list0, sens_0_label, target=target)
        test_dataset0 = UTKFaceRaw(test_list0, sens_0_label, target=target)
        thresh_dataset0 = UTKFaceRaw(thresh_list0, sens_0_label, target=target)
        train_dataset1 = UTKFaceRaw(train_list1, sens_1_label, target=target)
        val_dataset1 = UTKFaceRaw(val_list1, sens_1_label, target=target)
        test_dataset1 = UTKFaceRaw(test_list1, sens_1_label, target=target)
        thresh_dataset1 = UTKFaceRaw(thresh_list1, sens_1_label, target=target)

        train_loader0 = DataLoader(dataset=train_dataset0, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        val_loader0 = DataLoader(dataset=val_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        test_loader0 = DataLoader(dataset=test_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        thresh_loader0 = DataLoader(dataset=thresh_dataset0, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        train_loader1 = DataLoader(dataset=train_dataset1, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        val_loader1 = DataLoader(dataset=val_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        test_loader1 = DataLoader(dataset=test_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
        thresh_loader1 = DataLoader(dataset=thresh_dataset1, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)

        return train_loader0, val_loader0, test_loader0, thresh_loader0, train_loader1, val_loader1, test_loader1, thresh_loader1

	