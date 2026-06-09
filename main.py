import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
import numpy as np
import random

from registry import method_registry, dataset_registry

@hydra.main(config_path="conf", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    torch.backends.cudnn.benchmark = True

    dataset_name = cfg.datasets.dataset
    method_name = cfg.methods.method.type

    model_name = cfg.datasets.model_for[method_name]  
    model_cfg = OmegaConf.load(f"conf/models/{model_name}.yaml")

    with open_dict(cfg):
        cfg.models = model_cfg

    os.environ["WANDB_MEDIA_MAX_ITEMS"] = "0"

    data_root = "/path/to/your/data"
    savedir = "/path/to/your/save/directory"

    seed = cfg.seed * 5
    print('Currently working on - seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_class = dataset_registry[dataset_name]
    dataset = dataset_class(data_root)
    load_outputs = dataset.load(batch_size=cfg.datasets.batch_size, random_state=seed)

    train_loader0, val_loader0, test_loader0, thresh_loader0, train_loader1, val_loader1, test_loader1, thresh_loader1 = load_outputs
    for (x, y, z) in train_loader0:
        print(x.shape[1])

    context = {
        **cfg,
        "train_loader0": train_loader0,
        "train_loader1": train_loader1,
        "val_loader0": val_loader0,
        "val_loader1": val_loader1,
        "test_loader0": test_loader0,
        "test_loader1": test_loader1,
        "thresh_loader0": thresh_loader0,
        "thresh_loader1": thresh_loader1,
        "seed": seed,
    }

    method_class = method_registry[method_name]
    method = method_class(dataset_name, savedir, **context)

    if cfg.mode == "train":
        param0, param1 = method.train_step(**context)
        method.save_param(param0, param1, seed)
    
    method.infer_step(**context)

if __name__ == "__main__":
    main()