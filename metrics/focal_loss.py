import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, y):
        """
        y: ground truth
        logit: raw model output(logit)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logit, y, reduction='none')
        p = torch.sigmoid(logit)

        # p if y=1, 1-p othereise(y=0)
        eps = 1e-8
        p = torch.clamp(p, eps, 1 - eps)
        pt = y * p + (1-y) * (1-p)

        mod_factor = (1-pt) ** self.gamma

        # focal loss
        fl = mod_factor * BCE_loss

        return torch.mean(fl)