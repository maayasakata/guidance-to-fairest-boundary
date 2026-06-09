import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from models.mlp_film import FiLM_MLP

class ResNetWithFiLM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, n_layers, film_hidden_size, n_layers_film):
        super(ResNetWithFiLM, self).__init__()
        self.gen = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.gen.fc = nn.Linear(self.gen.fc.in_features, input_dim)
        self.cls = FiLM_MLP(input_dim, hidden_dim, num_classes, n_layers, film_hidden_size, n_layers_film) 

    def forward(self, x, lambda_reg=None):
        features = self.gen(x)
        output = self.cls(features, lambda_reg)
        return output