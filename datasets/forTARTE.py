import torch
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import typing as ty
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from datasets.my_tarte_preprocess import MyTARTE_TablePreprocessor


class TARTETrainingDataset(Dataset):
    def __init__(self, tarte_inputs: ty.Tuple[Tensor, Tensor, Tensor], labels: Tensor, sens_attrs: Tensor):
        self.tarte_inputs = tarte_inputs
        self.labels = labels
        self.sens_attrs = sens_attrs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        _, x, edge_attr, mask, _ = self.tarte_inputs[idx]
        return (x, edge_attr, mask), self.labels[idx], self.sens_attrs[idx]
    
def create_tarte_loaders(raw_df: pd.DataFrame, batch_size: int, num_workers: int, target: str, sens: str, sens_posi: str, seed=None):
    
    df = raw_df.reset_index(drop=True).copy()
    X = df.drop([target, sens], axis=1, errors="ignore")
    y = df[target].astype(float)
    a = df[sens]

    n = len(df)
    idx_all = np.arange(n)

    mask1 = (a == sens_posi)
    mask0 = ~mask1
    idx0 = idx_all[mask0]
    idx1 = idx_all[mask1]

    def split_indices(idxs, y_array):
        train_idx, test_idx = train_test_split(idxs, test_size=0.1, random_state=seed, stratify=y_array[idxs],)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed, stratify=y_array[train_idx],)
        train_idx, thresh_idx = train_test_split(train_idx, test_size=0.2, random_state=seed, stratify=y_array[train_idx],)
        return train_idx, val_idx, test_idx, thresh_idx

    y_array = y.values

    train0_idx, val0_idx, test0_idx, thresh0_idx = split_indices(idx0, y_array)
    train1_idx, val1_idx, test1_idx, thresh1_idx = split_indices(idx1, y_array)

    train_all_idx = np.concatenate([train0_idx, train1_idx])
    preprocessor = MyTARTE_TablePreprocessor()
    preprocessor.fit(X.iloc[train_all_idx], y.iloc[train_all_idx])
    preprocessor.y_ = None

    x_full = preprocessor.transform(X)  

    def make_dataset(idxs):
        xs = [x_full[i] for i in idxs]
        ys = torch.tensor(y_array[idxs], dtype=torch.float32)
        sens_vals = (a.values[idxs] == sens_posi).astype(float)
        sens_t = torch.tensor(sens_vals, dtype=torch.float32)
        return TARTETrainingDataset(xs, ys, sens_t)

    train_ds0 = make_dataset(train0_idx)
    val_ds0   = make_dataset(val0_idx)
    test_ds0  = make_dataset(test0_idx)
    thresh_ds0= make_dataset(thresh0_idx)

    train_ds1 = make_dataset(train1_idx)
    val_ds1   = make_dataset(val1_idx)
    test_ds1  = make_dataset(test1_idx)
    thresh_ds1= make_dataset(thresh1_idx)

    def make_loader(ds, shuffle, drop_last=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)

    train_loader0  = make_loader(train_ds0, shuffle=True, drop_last=False)
    val_loader0    = make_loader(val_ds0, shuffle=False)
    test_loader0   = make_loader(test_ds0, shuffle=False)
    thresh_loader0 = make_loader(thresh_ds0, shuffle=False)

    train_loader1  = make_loader(train_ds1, shuffle=True, drop_last=False)
    val_loader1    = make_loader(val_ds1, shuffle=False)
    test_loader1   = make_loader(test_ds1, shuffle=False)
    thresh_loader1 = make_loader(thresh_ds1, shuffle=False)

    return train_loader0, val_loader0, test_loader0, thresh_loader0, train_loader1, val_loader1, test_loader1, thresh_loader1,