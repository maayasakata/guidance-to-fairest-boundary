import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from models.mlp import MLPClassifier

class ResNetWithMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ResNetWithMLP, self).__init__()
        self.gen = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.gen.fc = nn.Linear(self.gen.fc.in_features, input_dim)
        self.cls = MLPClassifier(input_dim, hidden_dim, num_classes)

    def forward(self, x):
        features = self.gen(x)
        output = self.cls(features)
        return output