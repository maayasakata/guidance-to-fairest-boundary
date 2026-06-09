from abc import ABC
from hydra.utils import instantiate

class MethodBase(ABC):
    def __init__(self, dataset, savedir, **kwargs):
        self.dataset = dataset
        self.savedir = savedir
        self.epochs = kwargs["methods"]["epoch"]
        self.criterion = instantiate(kwargs["criterions"])
        self.device = kwargs["device"]
        self.folder = kwargs["methods"]["folder"]
        self.seed = kwargs["seed"]
        args_cfg = kwargs["models"]["args"]
        model0 = instantiate(args_cfg).to(self.device, non_blocking=True)
        model1 = instantiate(args_cfg).to(self.device, non_blocking=True)
    
        self.models = {"0": model0, "1": model1}
        self.optimizers = {
            "f": {
                "optimizer": instantiate(kwargs["methods"]["optimizers"]["f"]),
                "args": instantiate(kwargs["methods"]["optimizer_args"]["f"])
            }
        }
        if "g" in kwargs["methods"]["optimizers"]:
            self.optimizers["g"] = {
                "optimizer": instantiate(kwargs["methods"]["optimizers"]["g"]),
                "args": instantiate(kwargs["methods"]["optimizer_args"]["g"])
            }

    def process_x(self, x):
        if isinstance(x, (tuple, list)) and len(x) == 2:
            x[0] = x[0].to(self.device, non_blocking=True)
            x[1] = x[1].to(self.device, non_blocking=True)
        elif isinstance(x, (tuple, list)) and len(x) == 3:
            x_tarte, edge_attr, mask = x
            x_tarte = x_tarte.to(self.device, non_blocking=True)
            edge_attr = edge_attr.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)
            x = (x_tarte, edge_attr, mask)
        else:
            x = x.to(self.device, non_blocking=True)
        return x
    

    def train_step(self):
        raise NotImplementedError

    def infer_step(self):
        raise NotImplementedError

    def all_curve(self):
        raise NotImplementedError