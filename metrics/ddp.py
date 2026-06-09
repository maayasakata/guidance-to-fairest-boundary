import numpy as np

def DDP(logit0, logit1, tau0, tau1):
    ddp = np.mean(logit1 > tau1) - np.mean(logit0 > tau0)
    return ddp