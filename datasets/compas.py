import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.base import BaseLoader
from datasets.forTARTE import create_tarte_loaders

class CompasRaw(Dataset):
    def __init__(self, dataset):
        self.data = dataset#.sample(frac=0.01, random_state=42) 
        self.numerical_features = ["age", "juv_fel_count", "juv_misd_count", "juv_other_count", 'priors_count']
        self.categorical_features = ["sex", "age_cat", "race", "c_charge_degree"]
        self.target_column = 'two_year_recid'
        self.sensitive_column = "sex"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        x_num = torch.tensor(pd.to_numeric(data[self.numerical_features], errors='coerce').values, dtype=torch.float32)
        x_cat = torch.tensor(pd.to_numeric(data[self.categorical_features], errors='coerce').values, dtype=torch.long)
        label = torch.tensor(data[self.target_column], dtype=torch.float32)
        sens = torch.tensor(data[self.sensitive_column], dtype=torch.float32)
        return (x_num, x_cat), label, sens

class CompasLoader(BaseLoader):
    def load(self, batch_size=2048, random_state=None, model=None):
        data = pd.read_csv(f"{self.data_root}/compas/compas-scores-two-years.csv")
        data = data[(data['days_b_screening_arrest'] <= 30) &
                    (data['days_b_screening_arrest'] >= -30) &
                    (data['is_recid'] != -1) &
                    (data['c_charge_degree'] != "O") &
                    (data['score_text'] != "N/A")]
        
        data = data[["sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
                    "juv_other_count", "priors_count", "c_charge_degree", "two_year_recid"]].copy()
        
        rename_map = {
            "sex": "Sex",
            "age": "Age",
            "age_cat": "AgeCat",
            "race": "Race",
            "juv_fel_count": "JuvFelCount",
            "juv_misd_count": "JuvMisdCount",
            "juv_other_count": "JuvOtherCount",
            "priors_count": "PriorsCount",
            "c_charge_degree": "ChargeDegree",
            "two_year_recid": "Target",
        }
        data = data.rename(columns=rename_map)

        dtypes = {
            "Sex": "object",
            "Age": "float32",
            "AgeCat": "object",
            "Race": "object",
            "JuvFelCount": "float32",
            "JuvMisdCount": "float32",
            "JuvOtherCount": "float32",
            "PriorsCount": "float32",
            "ChargeDegree": "object",
            "Target": "int64",  
        }
        for col, dt in dtypes.items():
            data[col] = data[col].astype(dt)

        for col, dt in dtypes.items():
            if dt == "object":
                data[col] = data[col].astype(str).str.strip()

        return create_tarte_loaders(data, batch_size, self.num_workers, target="Target", sens="Race", sens_posi="Caucasian", seed=random_state,)
    

    