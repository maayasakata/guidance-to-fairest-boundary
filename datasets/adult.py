import torch
import pandas as pd
from torch.utils.data import Dataset
import os

from datasets.base import BaseLoader
from datasets.forTARTE import create_tarte_loaders

class AdultRaw(Dataset):
    def __init__(self, dataset):
        self.data = dataset
        self.numerical_features = ["Age", "Education-Num", "Capital Gain", "Capital Loss", "Hours per week"]
        self.categorical_features = ["Workclass", "Marital Status", "Occupation", "Relationship", "Race", "Sex", "Country"]
        self.target_column = "Target"
        self.sensitive_column = "Sex"
        self.categories = [self.data[col].max()+1 for col in self.categorical_features]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        x_num = torch.tensor(pd.to_numeric(data[self.numerical_features], errors='coerce').values, dtype=torch.float32)
        x_cat = torch.tensor(pd.to_numeric(data[self.categorical_features], errors='coerce').values, dtype=torch.long)
        label = torch.tensor(data[self.target_column], dtype=torch.float32)
        sens = torch.tensor(data[self.sensitive_column], dtype=torch.float32)

        return (x_num, x_cat), label, sens

class AdultLoader(BaseLoader):
    def load(self, batch_size=1024, random_state=None, model=None):
        data_root = os.path.join(self.data_root, "adult/")
        dtypes = [
            ("Age", "float32"), ("Workclass", "object"), ("fnlwgt", "float32"),
            ("Education", "object"), ("Education-Num", "float32"), ("Marital Status", "object"),
            ("Occupation", "object"), ("Relationship", "object"), ("Race", "object"),
            ("Sex", "object"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
            ("Hours per week", "float32"), ("Country", "object"), ("Target", "object")
        ]
        train_data = pd.read_csv(
            data_root+'adult.data',
            names=[d[0] for d in dtypes], 
            na_values="?",
            dtype=dict(dtypes)
        )
        test_data = pd.read_csv(
            data_root+'adult.test',
            skiprows=1,
            names=[d[0] for d in dtypes],
            na_values="?",
            dtype=dict(dtypes)
        )
        
        raw_all_data = pd.concat([train_data, test_data], ignore_index=True)
        for col, dt in dtypes:
            if dt == "object":
                raw_all_data[col] = raw_all_data[col].astype(str).str.strip()
        all_data = raw_all_data

        all_data["Target"] = all_data["Target"].str.contains(">50K").astype(int)
        
        return create_tarte_loaders(all_data, batch_size, self.num_workers, target="Target", sens="Sex", sens_posi="Male", seed=random_state)
    
