import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import typing as ty

from tarte_ai.tarte_model import TARTE_Base
from tarte_ai.tarte_utils import load_tarte_pretrain_model
from models.mlp import MLPClassifier


class MyTARTE_Base(TARTE_Base):
    def forward(self, inputs):
        x, edge_attr, mask = inputs
        return super().forward(x, edge_attr, mask)
    
class GeneratorWithTARTE(nn.Module):
    def __init__(self, tarte_base, layer_norm, d_out, configs, n_layers=5):
        super().__init__()
        
        self.tarte_base = tarte_base
        self.layer_norm = layer_norm 
        
        dim_t = configs['dim_transformer']

        if n_layers == 5:
            h1 = int(dim_t / 2)
            h2 = int(dim_t / 4)
            h3 = int(dim_t / 8)
            h4 = int(dim_t / 16)

            self.gen = nn.Sequential(
                nn.Linear(dim_t, h1),
                nn.ReLU(),
                nn.LayerNorm(h1),

                nn.Linear(h1, h2),
                nn.ReLU(),
                nn.LayerNorm(h2),

                nn.Linear(h2, h3),
                nn.ReLU(),
                nn.LayerNorm(h3),

                nn.Linear(h3, h4),
                nn.ReLU(),
                nn.LayerNorm(h4),

                nn.Linear(h4, d_out),
            )
        elif n_layers == 3:
            self.gen = nn.Sequential(
            nn.Linear(dim_t, int(dim_t / 2)),
            nn.ReLU(),
            nn.LayerNorm(int(dim_t / 2)),
            nn.Linear(int(dim_t / 2), int(dim_t / 4)),
            nn.ReLU(),
            nn.LayerNorm(int(dim_t / 4)),
            nn.Linear(int(dim_t / 4), d_out),
            )

        #--- TARTE Baseの凍結 ---#
        for param in self.tarte_base.parameters():
            param.requires_grad = False
        self.tarte_base.eval() 
        for param in self.layer_norm.parameters():
            param.requires_grad = False


    def forward(self, inputs):
        x = self.tarte_base(inputs) 
        x = self.layer_norm(x)
        cls_token_output = x[:, 0, :] 
        output = self.gen(cls_token_output)

        return output


class TARTEWithMLP(nn.Module):
    def __init__(self, d_out, num_classes, hidden_dim, n_layers):
        super(TARTEWithMLP, self).__init__()

        pretrain_weights, configs = load_tarte_pretrain_model()
        self.tarte_base = MyTARTE_Base(
            dim_input=configs['dim_input'],
            dim_transformer=configs['dim_transformer'],
            dim_feedforward=configs['dim_feedforward'],
            num_heads=configs['num_heads'],
            num_layers_transformer=configs['num_layers_transformer'],
            dropout=configs['dropout'],
        )

        base_weights = {
            k.replace('tarte_base.', ''): v 
            for k, v in pretrain_weights.items() 
            if k.startswith('tarte_base.')
        }
        self.tarte_base.load_state_dict(base_weights, strict=True)

        for param in self.tarte_base.parameters():
            param.requires_grad = False 
        self.tarte_base.eval()

        self.layer_norm = nn.LayerNorm(configs['dim_transformer'])

        self.gen = GeneratorWithTARTE(self.tarte_base, self.layer_norm, d_out, configs, n_layers=n_layers)

        self.cls = MLPClassifier(d_out, hidden_dim, num_classes)
        

    def forward(self, inputs: ty.Tuple[Tensor, ty.Optional[Tensor]]):
        x = self.gen(inputs)
        output = self.cls(x)
        
        return output