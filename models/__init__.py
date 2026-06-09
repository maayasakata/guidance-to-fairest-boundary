from .resnet_img import ResNetWithMLP
from .mlp import MLPClassifier

def build_model(model_cfg, split=False):
    model_cls = model_cfg["type"]
    model_args = model_cfg.get("args", {})
    return model_cls(**model_args)