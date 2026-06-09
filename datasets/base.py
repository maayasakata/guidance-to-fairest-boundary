from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

class BaseLoader(ABC):
    def __init__(self, data_root="/path/to/data/", num_workers=4):
        self.data_root = data_root
        self.num_workers = num_workers

    @abstractmethod
    def load(self):
        """Return the data loader."""
        pass

    def build_group_loaders(self, df0, df1, dataset_cls, dataset_kwargs: Optional[Dict[str, Any]] = None,
            *,
            test_size: float = 0.1, val_size: float = 0.1, thresh_size: float = 0.2, random_state: int = None,
            batch_size: int = 256, num_workers: int = 4, pin_memory: bool = True, prefetch_factor: int = 4,):
        
        dataset_kwargs = dataset_kwargs or {}

        # --- split group 0 ---
        train0, test0 = train_test_split(df0, test_size=test_size, random_state=random_state)
        train0, val0 = train_test_split(train0, test_size=val_size, random_state=random_state)
        train0, thresh0 = train_test_split(train0, test_size=thresh_size, random_state=random_state)
        # --- split group 1 ---
        train1, test1 = train_test_split(df1, test_size=test_size, random_state=random_state)
        train1, val1 = train_test_split(train1, test_size=val_size, random_state=random_state)
        train1, thresh1 = train_test_split(train1, test_size=thresh_size, random_state=random_state)

        # --- dataset instances ---
        ds_train0 = dataset_cls(dataset=train0, **dataset_kwargs)
        ds_val0   = dataset_cls(dataset=val0,   **dataset_kwargs)
        ds_test0  = dataset_cls(dataset=test0,  **dataset_kwargs)
        ds_thresh0 = dataset_cls(dataset=thresh0,  **dataset_kwargs)

        ds_train1 = dataset_cls(dataset=train1, **dataset_kwargs)
        ds_val1   = dataset_cls(dataset=val1,   **dataset_kwargs)
        ds_test1  = dataset_cls(dataset=test1,  **dataset_kwargs)
        ds_thresh1 = dataset_cls(dataset=thresh1,  **dataset_kwargs)

        def make_loader(ds, shuffle: bool):
            kwargs = dict(
                dataset=ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if num_workers and num_workers > 0:
                kwargs.update(
                    dict(
                        persistent_workers=True,
                        prefetch_factor=prefetch_factor,
                    )
                )
            return DataLoader(**kwargs)

        loaders = dict(
            train0  = make_loader(ds_train0,  True),
            val0    = make_loader(ds_val0,    False),
            test0   = make_loader(ds_test0,   False),
            thresh0 = make_loader(ds_thresh0, False),

            train1  = make_loader(ds_train1,  True),
            val1    = make_loader(ds_val1,    False),
            test1   = make_loader(ds_test1,   False),
            thresh1 = make_loader(ds_thresh1, False),
        )

        datasets = dict(
            train0=ds_train0, val0=ds_val0, test0=ds_test0, thresh0=ds_thresh0,
            train1=ds_train1, val1=ds_val1, test1=ds_test1, thresh1=ds_thresh1, 
        )

        frames = dict(
            train0=train0, val0=val0, test0=test0, thresh0=thresh0,
            train1=train1, val1=val1, test1=test1, thresh1=thresh1,
        )

        return loaders, datasets, frames