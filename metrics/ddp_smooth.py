import torch

class DDP_smooth:
    def __init__(self, k: float = 10.0):
        self.k = k

    def __call__(self, z0, z1, tau0, tau1):
        term1 = torch.sigmoid(self.k * (z1 - tau1)).mean()
        term0 = torch.sigmoid(self.k * (z0 - tau0)).mean()
        return torch.abs(term1 - term0)